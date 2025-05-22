# processor.py
from __future__ import annotations

from paddleocr import PaddleOCR

"""
Background processing loop for incoming frames.
Move your D-FINE / PaddleOCR inference and drawing here.
"""
# Server/apiBackup.py
import io
import queue
import threading
import time
from typing import Optional, Any
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from fastapi import HTTPException

# ---- local modules ----
from DFINE.tools.inference.torch_inf import YAMLConfig, OBJECTS365_CLASSES, draw
from Server.helper import videoHelper

# PROJECT_ROOT now points to your repo’s top-level folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---- model paths ----
MODEL_CFG = PROJECT_ROOT / "DFINE" / "configs" / "dfine" / "objects365" / "dfine_hgnetv2_x_obj365.yml"
MODEL_WEI = PROJECT_ROOT / "DFINE" / "model" / "dfine_x_obj365.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# global holder for the latest annotated JPEG
latest_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()

# ─── single-image inference that works on a PIL.Image (not a file-path) ────
_resize_to = (640, 640)
_tf = T.Compose([T.Resize(_resize_to), T.ToTensor()])


# ─── build & warm-load the D-FINE model exactly the way torch_inf does ─────
def build_dfine_model(cfg_path: str, ckpt_path: str, device: str = "cpu"):
    """Replicates the Model() logic from torch_inf, but returns a ready model."""
    cfg = YAMLConfig(cfg_path, resume=ckpt_path)

    # stop HGNetv2 layers from re-downloading ImageNet weights
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, imgs, orig_sizes):
            return self.post(self.model(imgs), orig_sizes)

    return DeployModel().to(device).eval()


model = build_dfine_model(str(MODEL_CFG), str(MODEL_WEI), DEVICE)

# Setup OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def torch_inf_pil(model, pil_img: Image.Image, device: str):
    """Return labels, boxes, scores → identical to detection.py output."""
    w, h = pil_img.size
    img_tensor = _tf(pil_img).unsqueeze(0).to(device)
    labels, boxes, scores = model(img_tensor, torch.tensor([[w, h]], device=device))
    return labels, boxes, scores


# Shared queue and lock should be passed in from server.py
def process_loop(frame_queue: queue.Queue[dict[str, Any]], jpeg_lock: threading.Lock, latest_jpeg_ref: dict) -> None:
    """
    Continuously pull raw JPEG bytes from frame_queue, decode, run inference,
    update latest_jpeg_ref['frame'] under jpeg_lock.
    """
    while True:
        frame_packet = frame_queue.get()
        if frame_packet is None:
            break

        frame_id = frame_packet["frame_id"]
        frame = frame_packet["data"]

        try:
            print(f"[Frame {frame_id:.3f}] timings:")
            t_start = time.perf_counter()
            # 1) Decode JPEG → BGR
            img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(400, "jpeg decode failed in processor")
            t2 = time.perf_counter()
            print(f"  ▸ JPEG decode      : {(t2 - t_start) * 1000:.1f} ms")

            # 2) Inference
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            labels, boxes, scores = torch_inf_pil(model, pil_img, DEVICE)
            t3 = time.perf_counter()
            print(f"  ▸ Inference        : {(t3 - t2) * 1000:.1f} ms")

            # 3) Draw and OCR results
            annotated = pil_img.copy()
            draw([annotated], labels, boxes, scores, thrh=0.5)

            # 1. Collect crops from all boxes
            cropped_images = []
            box_meta = []  # Store (x1, y1, ...) to match back after OCR
            for box, score in zip(boxes[0], scores[0]):
                if score < 0.4:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cropped_images.append(crop)
                box_meta.append((x1, y1, x2, y2))
            # H, W = img.shape[:2]
            #
            # for b, score in zip(boxes[0], scores[0]):
            #     if score < 0.4:
            #         continue
            #
            #     # ① pick the right format
            #     if b.max() <= 1.0:  # normalised
            #         cx, cy, w, h = b  # (cx, cy, w, h) – common in DETR
            #         x1 = int((cx - w / 2) * W)
            #         y1 = int((cy - h / 2) * H)
            #         x2 = int((cx + w / 2) * W)
            #         y2 = int((cy + h / 2) * H)
            #     else:  # already in pixels
            #         x1, y1, x2, y2 = map(int, b)
            #
            #     # ② clamp to the image
            #     x1, y1 = max(0, x1), max(0, y1)
            #     x2, y2 = min(W - 1, x2), min(H - 1, y2)
            #     if x2 <= x1 or y2 <= y1:
            #         continue
            #
            #     crop = img[y1:y2, x1:x2]
            #     cropped_images.append(crop)
            #     box_meta.append((x1, y1, x2, y2))

            # 2. Run OCR on the batch
            t_ocr_start = time.perf_counter()

            # batch one
            # results = ocr.ocr(cropped_images, det=False, cls=True)
            ocr_results = []
            for crop in cropped_images:
                res = ocr.ocr(crop, det=True, cls=True)  # single image
                ocr_results.append(res)
            t_ocr_end = time.perf_counter()
            print(f"  ▸ OCR (batch)      : {(t_ocr_end - t_ocr_start) * 1000:.1f} ms")

            # 3. Draw results back onto the original image
            # drawn = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            panel_rows = []

            # box_meta and ocr_results are 1-to-1
            for (x1, y1, x2, y2), ocr_lines in zip(box_meta, ocr_results):

                # 1) find the detector score that belongs to this box
                #    (boxes[0] is still torch; convert to int for comparison)
                score = None
                for b, s in zip(boxes[0], scores[0]):
                    bx1, by1, bx2, by2 = map(int, b.tolist())
                    if (bx1, by1, bx2, by2) == (x1, y1, x2, y2):
                        score = float(s)
                        break
                if score is None or score < 0.4:
                    continue

                row = {
                    "class_name": OBJECTS365_CLASSES[int(labels[0][0]) - 1][1],
                    "confidence": round(score, 3),
                    "box": [x1, y1, x2, y2]
                }

                # 2) attach OCR if present
                if ocr_lines:
                    ocr_results = []
                    for ocr_line in ocr_lines:
                        if ocr_line is None:
                            continue

                        for ocr_inside_line in ocr_line:
                            # Extract bounding box and (text, confidence) tuple
                            bbox, (text, confidence) = ocr_inside_line
                            ocr_results.append({
                                "ocr_text": text,
                                "ocr_conf": round(confidence, 3)  # Round the float, not the tuple
                            })

                    row["ocr_results"] = ocr_results

                panel_rows.append(row)

            # Convert back to PIL for final panel build
            composite = videoHelper.side_by_side(
                annotated,  # PIL frame
                videoHelper.build_detection_panel(panel_rows, annotated.height)
            )
            t4 = time.perf_counter()
            print(f"  ▸ Draw+Compose     : {(t4 - t3) * 1000:.1f} ms")

            # 4) Update latest_jpeg
            buf = io.BytesIO()
            composite.save(buf, format="JPEG", quality=85)
            with jpeg_lock:
                latest_jpeg_ref['frame'] = buf.getvalue()
            t5 = time.perf_counter()
            print(f"  ▸ JPEG re-encode   : {(t5 - t4) * 1000:.1f} ms")
            print(f"  ▸ Total            : {(t5 - t_start) * 1000:.1f} ms\n")

        finally:
            frame_queue.task_done()
