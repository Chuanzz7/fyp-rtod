import queue
import time
from multiprocessing import Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
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
    ocr = PaddleOCR(enable_hpi=True, use_gpu=True, use_angle_cls=True, lang="en", show_log=False)

    # Initialize DeepSort (tune as needed)
    tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)
    object_cache = {}
    frame_id = 0

    print("Inference & OCR buffers ready, waiting for frames...")

    frame_id = 0
    while True:
        try:
            frame_bytes = frame_input_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Convert bytes to numpy array of uint8
        processor_start = time.perf_counter()
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)

        # Decode JPEG to BGR image
        imgRaw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # img is now H x W x 3 in BGR

        if imgRaw is None:
            raise ValueError("Failed to decode JPEG image")

        # If you want RGB:
        img = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)

        output, inference_time = optimized_inference(ENGINE, _input_tensor, _orig_size, img)
        print(f"  ▸ Optimized Inference: {inference_time:.1f} ms")
        labels, boxes, scores = output["labels"], output["boxes"], output["scores"]

        # --------- Prepare detections for DeepSort ---------
        detections = []
        for i in range(boxes.shape[1]):
            score = scores[0, i].item()
            if score < 0.8:
                continue
            x1, y1, x2, y2 = [int(boxes[0, i, j].item()) for j in range(4)]
            label = int(labels[0, i].item())
            class_name = COCO_CLASSES[label][1] if 0 <= label < len(COCO_CLASSES) else "unknown"
            detections.append(([x1, y1, x2, y2], score, class_name))

        tracks = tracker.update_tracks(detections, frame=img)

        cropped_images, box_meta, panel_rows = [], [], []
        # --------- Extract crops & cache per DeepSort track ---------
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_name = track.det_class  # Correct attribute for class

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Save detection confidence score from track
            # track.det_conf is the detection confidence in deep_sort_realtime
            score = getattr(track, 'det_conf', None)
            if score is None:
                score = 1.0  # fallback if confidence not available

            cropped_images.append(crop)
            box_meta.append((x1, y1, x2, y2, track_id, class_name, score))
        # -----------------------------------------------------------

        # OCR each crop
        t_ocr_start = time.perf_counter()
        ocr_results = [ocr.ocr(crop, det=True, cls=True) for crop in cropped_images]
        t_ocr_end = time.perf_counter()
        print(f"  ▸ OCR (batch): {(t_ocr_end - t_ocr_start) * 1000:.1f} ms")

        sort_start = time.perf_counter()
        for (x1, y1, x2, y2, track_id, class_name, score), ocr_lines in zip(box_meta, ocr_results):
            # Store/update in RAM cache
            if track_id not in object_cache:
                object_cache[track_id] = {
                    "history": [],
                    "first_seen": frame_id,
                    "ocr_results": [],
                    "last_seen": frame_id,
                    "class_name": class_name,
                }
            object_cache[track_id]["history"].append((x1, y1, x2, y2, frame_id))
            object_cache[track_id]["last_seen"] = frame_id
            object_cache[track_id]["class_name"] = class_name

            # Filter OCR results by confidence threshold, e.g. 0.5
            conf_threshold = 0.5
            filtered_ocr = []
            if ocr_lines:
                for line in ocr_lines:
                    if line:
                        for _, (txt, conf) in line:
                            if conf >= conf_threshold:
                                filtered_ocr.append({"ocr_text": txt, "ocr_conf": round(conf, 3)})

            # Only update cache if currently empty to avoid overwriting good OCR
            if filtered_ocr and not object_cache[track_id]["ocr_results"]:
                object_cache[track_id]["ocr_results"] = filtered_ocr

            row = {
                "object_id": track_id,
                "class_name": class_name,
                "confidence": round(score, 3),
                "box": [x1, y1, x2, y2],
                "ocr_results": object_cache[track_id]["ocr_results"]
            }
            panel_rows.append(row)

        # Optional: Clean old tracks from cache
        max_disappear = 50
        remove_ids = [oid for oid, data in object_cache.items() if frame_id - data["last_seen"] > max_disappear]
        for oid in remove_ids:
            del object_cache[oid]
        sort_end = time.perf_counter()
        print(f"  ▸ SORT: {(sort_end - sort_start) * 1000:.1f} ms")

        draw_start = time.perf_counter()
        pil_img = draw([Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))], labels, boxes, scores, 0.8)
        draw_end = time.perf_counter()
        print(f"  ▸ Draw: {(draw_end - draw_start) * 1000:.1f} ms")

        data = {
            "frame_id": frame_id,
            "img": img,
            "pil_img": pil_img[0],
            "panel_rows": panel_rows,
        }

        processor_end = time.perf_counter()
        print(f"Processor End: {(processor_end - processor_start) * 1000:.1f} ms")
        output_queue.put(data)
        frame_id += 1
