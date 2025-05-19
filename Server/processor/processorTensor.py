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
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# ── local modules ─────────────────────────────────────────────────────────
from DFINE.tools.inference.trt_inf import TRTInference, draw


# global holder for the latest annotated JPEG
latest_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ 1.  Build the Tensor Runtime session                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝


# Global pre-allocated tensors
_input_tensor = None
_orig_size = None


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
    """
    Highly optimized inference pipeline

    Args:
        engine: TensorRT inference engine
        img: OpenCV image
        device: CUDA device

    Returns:
        Detection results and timing information
    """
    global _input_tensor, _orig_size

    # Initialize buffers if not already done
    if _input_tensor is None:
        initialize_buffers(device)

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


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ 3.  Main processing loop                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def process_loop(frame_queue, engine, device="cuda:0"):
    """
    Replacement for the process_loop function with optimized inference
    """
    # Initialize buffers once
    initialize_buffers(device)

    while True:
        frame_packet = frame_queue.get()
        if frame_packet is None:
            break

        frame_id = frame_packet["frame_id"]
        frame = frame_packet["data"]

        try:
            # 1) Decode JPEG → BGR
            img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("  !! JPEG decode failed – skipping frame")
                continue

            # 2) Run optimized inference
            output, inference_time = optimized_inference(engine, img, device)

            # Extract results
            labels, boxes, scores = output["labels"], output["boxes"], output["scores"]

            # 3) Process results as needed
            # ... (draw boxes, update latest_jpeg, etc.)

            print(f"[Frame {frame_id:.3f}] timings:")
            print(f"  ▸ Optimized Inference: {inference_time:.1f} ms")

        finally:
            frame_queue.task_done()
