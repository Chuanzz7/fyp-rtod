import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request, Response

# ── local modules ─────────────────────────────────────────────────────────
from DFINE.tools.inference.trt_inf import TRTInference

# ── paths & constants ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSOR_MODEL = PROJECT_ROOT / "DFINE" / "model" / "model.engine"
DEVICE = "cuda:0"
ENGINE = TRTInference(TENSOR_MODEL)

# Global pre-allocated tensors
_input_tensor = None
_orig_size = None
app = FastAPI(title="D‑FINE Object Detection API")


def initialize_buffers(device="cuda:0"):
    """Initialize global buffers once at startup"""
    global _input_tensor, _orig_size

    # Make sure device is properly handled
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


initialize_buffers()


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


def optimized_inference(engine, img, device="cuda:0"):
    global _input_tensor, _orig_size
    if _input_tensor is None:
        initialize_buffers(device)
    h, w = img.shape[:2]

    t_start = time.perf_counter()
    # Preprocessing
    t0 = time.perf_counter()
    fast_preprocess(img, _input_tensor)
    _orig_size[0, 0] = w
    _orig_size[0, 1] = h
    t1 = time.perf_counter()

    # Inference
    blob = {
        "images": _input_tensor,
        "orig_target_sizes": _orig_size,
    }
    output = engine(blob)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    preprocess_time = (t1 - t0) * 1000
    inference_time = (t2 - t1) * 1000
    total_time = (t2 - t_start) * 1000

    return output, preprocess_time, inference_time, total_time


@app.post("/upload_frame")
async def upload_frame(request: Request):
    data = await request.body()
    # 1) Decode JPEG → BGR
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print("  !! JPEG decode failed – skipping frame")

    # 2) Run optimized inference
    output, preprocess_time, inference_time, total_time = optimized_inference(ENGINE, img, DEVICE)

    # Extract results
    labels, boxes, scores = output["labels"], output["boxes"], output["scores"]

    print(f"Preprocess: {preprocess_time:.2f} ms | Inference: {inference_time:.2f} ms | Total: {total_time:.2f} ms")
    return Response(status_code=200)


# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
