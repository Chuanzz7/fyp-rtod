import queue
import time
from multiprocessing import Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from paddleocr import PaddleOCR

from DFINE.tools.inference.trt_inf import TRTInference, draw
from Server.helper.dataClass import COCO_CLASSES

# ── paths & constants ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TENSOR_MODEL = PROJECT_ROOT / "DFINE" / "model" / "model.engine"
DEVICE = "cuda:0"


def initialize_buffers(device="cuda:0"):
    global _input_tensor, _orig_size
    if isinstance(device, str):
        device = torch.device(device)
    _input_tensor = torch.empty((1, 3, 640, 640), dtype=torch.float32, device=device)
    _orig_size = torch.empty((1, 2), dtype=torch.int32, device=device)

    # Warm up GPU
    dummy_data = torch.zeros_like(_input_tensor)
    for _ in range(10):
        _ = dummy_data + 1.0
    torch.cuda.synchronize()
    print("Buffers initialized and GPU warmed up")


def fast_preprocess(img, out_tensor=None):
    """
    Fast image preprocessing directly with OpenCV and torch

    Args:
        img: OpenCV image (BGR)
        out_tensor: Pre-allocated output tensor

    Returns:
        Preprocessed tensor
    """
    # Resize with OpenCV (much faster than PIL)
    resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1] and convert to tensor
    resized = resized.astype(np.float32) / 255.0

    # Handle output tensor
    if out_tensor is None:
        # Create new tensor if none provided
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    else:
        # Reuse existing tensor
        out_tensor[0] = torch.from_numpy(resized).permute(2, 0, 1)
        return out_tensor


def optimized_inference(engine, _input_tensor, _orig_size, img, ):
    """
    Highly optimized inference pipeline

    Args:
        engine: TensorRT inference engine
        img: OpenCV image
        device: CUDA device

    Returns:
        Detection results and timing information
    """

    h, w = img.shape[:2]

    # Start timing
    t_start = time.perf_counter()

    # Fast preprocess directly to pre-allocated tensor
    fast_preprocess(img, _input_tensor)

    # Set original size
    _orig_size[0, 0] = w
    _orig_size[0, 1] = h

    # Create blob with pre-allocated tensors
    blob = {
        "images": _input_tensor,
        "orig_target_sizes": _orig_size,
    }

    # Run inference
    output = engine(blob)

    # Make sure GPU is done
    torch.cuda.synchronize()

    # End timing
    t_end = time.perf_counter()
    inference_time = (t_end - t_start) * 1000

    return output, inference_time


def processor_tensor_main(frame_input_queue: Queue, output_queue: Queue):
    initialize_buffers()
    ENGINE = TRTInference(TENSOR_MODEL)
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    print("Inference & OCR buffers ready, waiting for frames...")

    frame_id = 0
    while True:
        try:
            frame = frame_input_queue.get(timeout=1)
        except queue.Empty:
            continue

        img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("JPEG decode failed – skipping frame")
            continue

        output, inference_time = optimized_inference(ENGINE, _input_tensor, _orig_size, img)
        print(f"  ▸ Optimized Inference: {inference_time:.1f} ms")
        labels, boxes, scores = output["labels"], output["boxes"], output["scores"]

        cropped_images, box_meta, panel_rows = [], [], []
        for i in range(boxes.shape[1]):  # boxes: [1, N, 4]
            score = scores[0, i].item()
            if score < 0.8:
                continue
            x1, y1, x2, y2 = [int(boxes[0, i, j].item()) for j in range(4)]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cropped_images.append(crop)
            box_meta.append((x1, y1, x2, y2))

        # OCR each crop
        t_ocr_start = time.perf_counter()
        ocr_results = [ocr.ocr(crop, det=True, cls=True) for crop in cropped_images]
        t_ocr_end = time.perf_counter()
        print(f"OCR (batch): {(t_ocr_end - t_ocr_start) * 1000:.1f} ms")

        for (x1, y1, x2, y2), ocr_lines, i in zip(box_meta, ocr_results, range(len(box_meta))):
            score = scores[0, i].item()
            label = int(labels[0, i].item())
            if score < 0.4:
                continue

            row = {
                "class_name": COCO_CLASSES[label][1] if 0 <= label < len(COCO_CLASSES) else "unknown",
                "confidence": round(score, 3),
                "box": [x1, y1, x2, y2],
            }
            if ocr_lines:
                row["ocr_results"] = [
                    {"ocr_text": txt, "ocr_conf": round(conf, 3)}
                    for line in ocr_lines if line
                    for _, (txt, conf) in line
                ]
            panel_rows.append(row)

        pil_img = draw([Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))], labels, boxes, scores, 0.8)
        data = {
            "frame_id": frame_id,
            "img": img,
            "pil_img": pil_img[0],
            "panel_rows": panel_rows,
        }

        output_queue.put(data)
        frame_id += 1