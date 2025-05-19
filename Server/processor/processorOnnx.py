# processor.py
from __future__ import annotations

"""
Main background worker that pulls JPEG frames from the queue, runs
object‑detection (D‑FINE exported to ONNX) and OCR (PaddleOCR), draws the
results and stores the latest composite as a JPEG byte‑string that the
FastAPI endpoint can return on demand.

2025‑05‑18 • patch‑02
─────────────────────
Fixes the onnxruntime ``INVALID_ARGUMENT: Unexpected input data type …
expected: (tensor(int64))`` error.

Cause: we were always sending ``orig_target_sizes`` as *float32* when the
exported graph actually declares that input as *int64*.

Fix: look at the **TensorProto** type string and build the NumPy array with
``dtype=np.int64`` whenever the graph says so (and default to *float32*
otherwise).
"""

# ── stdlib & third‑party ──────────────────────────────────────────────────
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from paddleocr import PaddleOCR
from PIL import Image

# ── local modules ─────────────────────────────────────────────────────────
from DFINE.tools.inference.torch_inf import OBJECTS365_CLASSES
from DFINE.tools.inference.onnx_inf import draw, process_image, resize_with_aspect_ratio

# ── paths & constants ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONNX_MODEL = PROJECT_ROOT / "DFINE" / "model" / "dfine_x_coco.onnx"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# global holder for the latest annotated JPEG
latest_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()

# ── preprocess op shared by all inference calls ──────────────────────────
_resize_to = (640, 640)
_tf = T.Compose([T.Resize(_resize_to), T.ToTensor()])


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ 1.  Build the ONNX Runtime session                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def build_onnx_session(onnx_path: str, device: str = "cpu") -> ort.InferenceSession:
    """Return an ONNX Runtime session with the best provider available."""
    providers = ["CUDAExecutionProvider",
                 "CPUExecutionProvider",]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def process_image(sess, im_pil):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    im_data = transforms(resized_im_pil).unsqueeze(0)
    t2 = time.perf_counter()
    output = sess.run(
        output_names=None,
        input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
    )
    t3 = time.perf_counter()
    print(f"  ▸ Model Inference        : {(t3 - t2) * 1000:.1f} ms")
    labels, boxes, scores = output
    return [im_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)]


session = build_onnx_session(str(ONNX_MODEL), DEVICE)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ 2.  PaddleOCR – silent by default                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ 3.  ONNX single‑image helper                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def _dtype_from_ort(info: ort.NodeArg) -> np.dtype:
    """Map ORT's type string (e.g. "tensor(int64)") to a NumPy dtype."""
    return np.int64 if "int64" in info.type else np.float32


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ 4.  Main processing loop                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def process_loop(frame_queue: queue.Queue[dict[str, Any]],
                 jpeg_lock: threading.Lock,
                 latest_jpeg_ref: dict) -> None:
    """Continuously pulls JPEG bytes → runs detection + OCR → stores composite."""
    while True:
        frame_packet = frame_queue.get()
        if frame_packet is None:
            break

        frame_id: int | float = frame_packet["frame_id"]
        frame: bytes = frame_packet["data"]

        try:
            print(f"[Frame {frame_id:.3f}] timings:")
            t_start = time.perf_counter()

            # 1) Decode JPEG → BGR
            img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("  !! JPEG decode failed – skipping frame")
                continue
            t2 = time.perf_counter()
            print(f"  ▸ JPEG decode      : {(t2 - t_start) * 1000:.1f} ms")

            # 2) Detection
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            real_inference = time.perf_counter()
            annotated, labels, boxes, scores, ratio, paddings = process_image(session, pil_img)
            end_inference = time.perf_counter()
            print(f"  ▸ Inference        : {(end_inference - real_inference) * 1000:.1f} ms")

            # 3) Draw detection boxes
            draw([pil_img], labels, boxes, scores, ratio, paddings, thrh=0.5)

            # 4) Prepare crops for OCR
            cropped_images, box_meta = [], []
            for box, score in zip(boxes[0], scores[0]):
                if score < 0.4:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cropped_images.append(crop)
                box_meta.append((x1, y1, x2, y2))

            # 5) OCR each crop
            t_ocr_start = time.perf_counter()
            ocr_results = [ocr.ocr(crop, det=True, cls=True) for crop in cropped_images]
            t_ocr_end = time.perf_counter()
            print(f"  ▸ OCR (batch)      : {(t_ocr_end - t_ocr_start) * 1000:.1f} ms")

            # 6) Build side panel rows
            panel_rows = []
            for (x1, y1, x2, y2), ocr_lines in zip(box_meta, ocr_results):
                score = None
                for b, s in zip(boxes[0], scores[0]):
                    if tuple(map(int, b.tolist())) == (x1, y1, x2, y2):
                        score = float(s)
                        break
                if score is None or score < 0.4:
                    continue

                row = {
                    "class_name": OBJECTS365_CLASSES[int(labels[0][0]) - 1][1],
                    "confidence": round(score, 3),
                    "box": [x1, y1, x2, y2],
                }

                if ocr_lines:
                    row["ocr_results"] = [
                        {"ocr_text": txt, "ocr_conf": round(conf, 3)}
                        for line in ocr_lines if line
                        for _, (txt, conf) in line
                    ]
                    print(row["ocr_results"])
                panel_rows.append(row)

            # # 7) Composite frame + panel
            # composite = videoHelper.side_by_side(
            #     pil_img,  # PIL image
            #     videoHelper.build_detection_panel(panel_rows, pil_img.height)
            # )
            # t4 = time.perf_counter()
            # print(f"  ▸ Draw+Compose     : {(t4 - t3) * 1000:.1f} ms")
            #
            # # 4) Update latest_jpeg
            # buf = io.BytesIO()
            # composite.save(buf, format="JPEG", quality=85)
            # with jpeg_lock:
            #     latest_jpeg_ref['frame'] = buf.getvalue()
            # t5 = time.perf_counter()
            # print(f"  ▸ JPEG re-encode   : {(t5 - t4) * 1000:.1f} ms")
            print(f"  ▸ Total            : {(t_ocr_end - t_start) * 1000:.1f} ms\n")

        finally:
            frame_queue.task_done()
