# # processor.py
# from __future__ import annotations
#
# import io
# import queue
# import threading
# from typing import Any
#
# from PIL import Image
# from paddleocr import PaddleOCR
#
# from DFINE.detection import OBJECTS365_CLASSES
# from Server.helper import videoHelper
#
# """
# Main background worker that pulls JPEG frames from the queue, runs
# object‑detection (D‑FINE exported to ONNX) and OCR (PaddleOCR), draws the
# results and stores the latest composite as a JPEG byte‑string that the
# FastAPI endpoint can return on demand.
#
# 2025‑05‑18 • patch‑02
# ─────────────────────
# Fixes the onnxruntime ``INVALID_ARGUMENT: Unexpected input data type …
# expected: (tensor(int64))`` error.
#
# Cause: we were always sending ``orig_target_sizes`` as *float32* when the
# exported graph actually declares that input as *int64*.
#
# Fix: look at the **TensorProto** type string and build the NumPy array with
# ``dtype=np.int64`` whenever the graph says so (and default to *float32*
# otherwise).
# """
#
# # ── stdlib & third‑party ──────────────────────────────────────────────────
# import time
#
# import cv2
# import numpy as np
# import torch
#
# # ── local modules ─────────────────────────────────────────────────────────
# from DFINE.tools.inference.trt_inf import draw
#
#
# # ╔═══════════════════════════════════════════════════════════════════════╗
# # ║ 1.  Build the Tensor Runtime session                                    ║
# # ╚═══════════════════════════════════════════════════════════════════════╝
#
#
# def fast_preprocess(img, out_tensor=None):
#     """
#     Fast image preprocessing directly with OpenCV and torch
#
#     Args:
#         img: OpenCV image (BGR)
#         out_tensor: Pre-allocated output tensor
#
#     Returns:
#         Preprocessed tensor
#     """
#     # Resize with OpenCV (much faster than PIL)
#     resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
#
#     # Convert BGR to RGB
#     resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#
#     # Normalize to [0,1] and convert to tensor
#     resized = resized.astype(np.float32) / 255.0
#
#     # Handle output tensor
#     if out_tensor is None:
#         # Create new tensor if none provided
#         return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
#     else:
#         # Reuse existing tensor
#         out_tensor[0] = torch.from_numpy(resized).permute(2, 0, 1)
#         return out_tensor
#
#
# def optimized_inference(engine, _input_tensor, _orig_size, img, ):
#     """
#     Highly optimized inference pipeline
#
#     Args:
#         engine: TensorRT inference engine
#         img: OpenCV image
#         device: CUDA device
#
#     Returns:
#         Detection results and timing information
#     """
#
#     h, w = img.shape[:2]
#
#     # Start timing
#     t_start = time.perf_counter()
#
#     # Fast preprocess directly to pre-allocated tensor
#     fast_preprocess(img, _input_tensor)
#
#     # Set original size
#     _orig_size[0, 0] = w
#     _orig_size[0, 1] = h
#
#     # Create blob with pre-allocated tensors
#     blob = {
#         "images": _input_tensor,
#         "orig_target_sizes": _orig_size,
#     }
#
#     # Run inference
#     output = engine(blob)
#
#     # Make sure GPU is done
#     torch.cuda.synchronize()
#
#     # End timing
#     t_end = time.perf_counter()
#     inference_time = (t_end - t_start) * 1000
#
#     return output, inference_time
#
#
# # ╔═══════════════════════════════════════════════════════════════════════╗
# # ║ 2.  PaddleOCR – silent by default                                     ║
# # ╚═══════════════════════════════════════════════════════════════════════╝
#
# ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
#
#
# # ╔═══════════════════════════════════════════════════════════════════════╗
# # ║ 3.  Main processing loop                                              ║
# # ╚═══════════════════════════════════════════════════════════════════════╝
#
#
# def inference_thread(
#         frame_queue: "queue.Queue[dict[str, Any]]",
#         engine,
#         _input_tensor,
#         _orig_size,
#         jpeg_lock: threading.Lock,
#         latest_jpeg_ref: dict,
# ):
#     while True:
#         frame_packet = frame_queue.get()
#         if frame_packet is None:
#             break
#
#         frame_id = frame_packet["frame_id"]
#         frame = frame_packet["data"]
#
#         try:
#             t_start = time.perf_counter()
#
#             # 1) Decode JPEG → BGR
#             img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
#             if img is None:
#                 print("  !! JPEG decode failed – skipping frame")
#                 continue
#
#             # 2) Run optimized inference
#             output, inference_time = optimized_inference(engine, _input_tensor, _orig_size, img)
#
#             # Extract results
#             labels, boxes, scores = output["labels"], output["boxes"], output["scores"]
#
#             # 3) Process results as needed
#             # ... (draw boxes, update latest_jpeg, etc.)
#
#             print(f"[Frame {frame_id:.3f}] timings:")
#             print(f"  ▸ Optimized Inference: {inference_time:.1f} ms")
#             pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#             # 3) Draw detection boxes
#             draw([pil_img], labels, boxes, scores, thrh=0.5)
#
#             # 4) Prepare crops for OCR
#             cropped_images, box_meta = [], []
#             for box, score in zip(boxes[0], scores[0]):
#                 if score < 0.4:
#                     continue
#                 x1, y1, x2, y2 = map(int, box.tolist())
#                 crop = img[y1:y2, x1:x2]
#                 if crop.size == 0:
#                     continue
#                 cropped_images.append(crop)
#                 box_meta.append((x1, y1, x2, y2))
#
#             # 5) OCR each crop
#             t_ocr_start = time.perf_counter()
#             ocr_results = [ocr.ocr(crop, det=True, cls=True) for crop in cropped_images]
#             t_ocr_end = time.perf_counter()
#             print(f"  ▸ OCR (batch)      : {(t_ocr_end - t_ocr_start) * 1000:.1f} ms")
#
#             # 6) Build side panel rows
#             panel_rows = []
#             for (x1, y1, x2, y2), ocr_lines in zip(box_meta, ocr_results):
#                 score = None
#                 for b, s in zip(boxes[0], scores[0]):
#                     if tuple(map(int, b.tolist())) == (x1, y1, x2, y2):
#                         score = float(s)
#                         break
#                 if score is None or score < 0.4:
#                     continue
#
#                 row = {
#                     "class_name": OBJECTS365_CLASSES[int(labels[0][0]) - 1][1],
#                     "confidence": round(score, 3),
#                     "box": [x1, y1, x2, y2],
#                 }
#
#                 if ocr_lines:
#                     row["ocr_results"] = [
#                         {"ocr_text": txt, "ocr_conf": round(conf, 3)}
#                         for line in ocr_lines if line
#                         for _, (txt, conf) in line
#                     ]
#                     print(row["ocr_results"])
#                 panel_rows.append(row)
#             t3 = time.perf_counter()
#
#             # 7) Composite frame + panel
#             composite = videoHelper.side_by_side(
#                 pil_img,  # PIL image
#                 videoHelper.build_detection_panel(panel_rows, pil_img.height)
#             )
#             t4 = time.perf_counter()
#             print(f"  ▸ Draw+Compose     : {(t4 - t3) * 1000:.1f} ms")
#
#             # 4) Update latest_jpeg
#             buf = io.BytesIO()
#             composite.save(buf, format="JPEG", quality=85)
#             with jpeg_lock:
#                 latest_jpeg_ref['frame'] = buf.getvalue()
#             t5 = time.perf_counter()
#             print(f"  ▸ JPEG re-encode   : {(t5 - t4) * 1000:.1f} ms")
#             print(f"  ▸ Total            : {(t_ocr_end - t_start) * 1000:.1f} ms\n")
#
#         finally:
#             frame_queue.task_done()
