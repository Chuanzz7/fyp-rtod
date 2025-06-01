import queue
import time
from multiprocessing import Queue
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from paddleocr import PaddleOCR

from DFINE.tools.inference.trt_inf import TRTInference, draw
from Detection.helper.dataClass import COCO_CLASSES

# ── Constants ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TENSOR_MODEL = PROJECT_ROOT / "DFINE" / "model" / "model.engine"
DEVICE = "cuda:0"
OCR_CONFIDENCE_THRESHOLD = 0.5
DETECTION_CONFIDENCE_THRESHOLD = 0.8
MAX_TRACK_DISAPPEAR_FRAMES = 50

# Global buffers
_input_tensor = None
_orig_size = None


class GPUBufferManager:
    """Manages GPU memory buffers for inference"""

    @staticmethod
    def initialize_buffers(device: str = "cuda:0") -> None:
        """Initialize GPU buffers and warm up"""
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
        print("GPU buffers initialized and warmed up")


class ImagePreprocessor:
    """Handles fast image preprocessing"""

    @staticmethod
    def preprocess(img: np.ndarray, out_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fast image preprocessing with OpenCV and torch"""
        # Resize with OpenCV (faster than PIL)
        resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1]
        resized = resized.astype(np.float32) / 255.0

        if out_tensor is None:
            return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        else:
            out_tensor[0] = torch.from_numpy(resized).permute(2, 0, 1)
            return out_tensor


class InferenceEngine:
    """Handles optimized TensorRT inference"""

    def __init__(self, model_path: Path):
        self.engine = TRTInference(model_path)

    def run_inference(self, img: np.ndarray) -> Tuple[Dict[str, torch.Tensor], float]:
        """Run optimized inference pipeline"""
        h, w = img.shape[:2]
        t_start = time.perf_counter()

        # Preprocess to pre-allocated tensor
        ImagePreprocessor.preprocess(img, _input_tensor)

        # Set original size
        _orig_size[0, 0] = w
        _orig_size[0, 1] = h

        # Create blob
        blob = {
            "images": _input_tensor,
            "orig_target_sizes": _orig_size,
        }

        # Run inference
        output = self.engine(blob)
        torch.cuda.synchronize()

        t_end = time.perf_counter()
        inference_time = (t_end - t_start) * 1000

        return output, inference_time


class DetectionProcessor:
    """Processes detection results and prepares them for tracking"""

    @staticmethod
    def process_detections(output: Dict[str, torch.Tensor],
                           confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD) -> List[Tuple]:
        """Convert model output to detection format for DeepSort"""
        labels, boxes, scores = output["labels"], output["boxes"], output["scores"]
        detections = []

        for i in range(boxes.shape[1]):
            score = scores[0, i].item()
            if score < confidence_threshold:
                continue

            x1, y1, x2, y2 = [int(boxes[0, i, j].item()) for j in range(4)]
            label = int(labels[0, i].item())
            class_name = COCO_CLASSES[label][1] if 0 <= label < len(COCO_CLASSES) else "unknown"
            detections.append(([x1, y1, x2, y2], score, class_name))

        return detections


class TrackingManager:
    """Manages object tracking with DeepSort"""

    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)

    def update_tracks(self, detections: List[Tuple], frame: np.ndarray) -> List:
        """Update tracks with new detections"""
        return self.tracker.update_tracks(detections, frame=frame)

    @staticmethod
    def extract_track_info(tracks: List) -> List[Tuple]:
        """Extract confirmed track information"""
        track_info = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_name = track.det_class
            score = getattr(track, 'det_conf', 1.0)

            track_info.append((x1, y1, x2, y2, track_id, class_name, score))

        return track_info


class OCRProcessor:
    """Handles OCR processing with caching"""

    def __init__(self):
        self.ocr = PaddleOCR(
            enable_hpi=True,
            use_gpu=True,
            use_angle_cls=True,
            lang="en",
            show_log=False
        )

    def process_crops(self, crops: List[np.ndarray]) -> List:
        """Process multiple image crops with OCR"""
        if not crops:
            return []

        t_start = time.perf_counter()
        results = [self.ocr.ocr(crop, det=True, cls=True) for crop in crops]
        t_end = time.perf_counter()

        print(f"  ▸ OCR (batch): {(t_end - t_start) * 1000:.1f} ms")
        return results

    @staticmethod
    def filter_ocr_results(ocr_lines: List,
                           confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD) -> List[Dict]:
        """Filter OCR results by confidence threshold"""
        filtered_ocr = []

        if ocr_lines:
            for line in ocr_lines:
                if line:
                    for _, (txt, conf) in line:
                        if conf >= confidence_threshold:
                            filtered_ocr.append({
                                "ocr_text": txt,
                                "ocr_conf": round(conf, 3)
                            })

        return filtered_ocr


class ObjectCache:
    """Manages object tracking cache and OCR results"""

    def __init__(self):
        self.cache: Dict[int, Dict[str, Any]] = {}

    def update_object(self, track_id: int, box: Tuple[int, int, int, int],
                      frame_id: int, class_name: str, ocr_results: List[Dict]) -> None:
        """Update object in cache"""
        x1, y1, x2, y2 = box

        if track_id not in self.cache:
            self.cache[track_id] = {
                "history": [],
                "first_seen": frame_id,
                "ocr_results": [],
                "last_seen": frame_id,
                "class_name": class_name,
                "ocr_processed": False  # Flag to track if OCR has been done
            }

        # Update tracking info
        self.cache[track_id]["history"].append((x1, y1, x2, y2, frame_id))
        self.cache[track_id]["last_seen"] = frame_id
        self.cache[track_id]["class_name"] = class_name

        # Only update OCR if we haven't processed it yet and we have new results
        if not self.cache[track_id]["ocr_processed"] and ocr_results:
            self.cache[track_id]["ocr_results"] = ocr_results
            self.cache[track_id]["ocr_processed"] = True
            print(f"  ▸ OCR saved for track {track_id}: {len(ocr_results)} results")

    def needs_ocr(self, track_id: int) -> bool:
        """Check if OCR processing is needed for this track"""
        if track_id not in self.cache:
            return True
        return not self.cache[track_id]["ocr_processed"]

    def get_ocr_results(self, track_id: int) -> List[Dict]:
        """Get cached OCR results for a track"""
        if track_id in self.cache:
            return self.cache[track_id]["ocr_results"]
        return []

    def cleanup_old_tracks(self, current_frame: int,
                           max_disappear: int = MAX_TRACK_DISAPPEAR_FRAMES) -> None:
        """Remove old tracks from cache"""
        remove_ids = [
            oid for oid, data in self.cache.items()
            if current_frame - data["last_seen"] > max_disappear
        ]

        for oid in remove_ids:
            del self.cache[oid]
            print(f"  ▸ Removed old track {oid} from cache")


class CropExtractor:
    """Extracts image crops from tracked objects"""

    @staticmethod
    def extract_crops(img: np.ndarray, track_info: List[Tuple],
                      object_cache: ObjectCache) -> Tuple[List[np.ndarray], List[Tuple], List[int]]:
        """Extract crops for objects that need OCR processing"""
        crops = []
        crop_metadata = []
        track_ids_needing_ocr = []

        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Only add to OCR queue if OCR hasn't been processed yet
            if object_cache.needs_ocr(track_id):
                crops.append(crop)
                crop_metadata.append((x1, y1, x2, y2, track_id, class_name, score))
                track_ids_needing_ocr.append(track_id)

        return crops, crop_metadata, track_ids_needing_ocr


class ResultAssembler:
    """Assembles final results for output"""

    @staticmethod
    def create_panel_rows(track_info: List[Tuple], object_cache: ObjectCache) -> List[Dict]:
        """Create panel rows with all tracking and OCR information"""
        panel_rows = []

        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            row = {
                "object_id": track_id,
                "class_name": class_name,
                "confidence": round(float(score or 1.0), 3),
                "box": [x1, y1, x2, y2],
                "ocr_results": object_cache.get_ocr_results(track_id)
            }
            panel_rows.append(row)

        return panel_rows


def decode_frame(frame_bytes: bytes) -> np.ndarray:
    """Decode JPEG frame bytes to OpenCV image"""
    np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_raw is None:
        raise ValueError("Failed to decode JPEG image")

    return cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)


def processor_tensor_main(frame_input_queue: Queue, output_queue: Queue):
    """Main processing function with refactored components"""
    # Initialize components
    GPUBufferManager.initialize_buffers()
    inference_engine = InferenceEngine(TENSOR_MODEL)
    tracking_manager = TrackingManager()
    ocr_processor = OCRProcessor()
    object_cache = ObjectCache()

    print("All components initialized, waiting for frames...")

    frame_id = 0
    while True:
        try:
            frame_bytes = frame_input_queue.get(timeout=1)
        except queue.Empty:
            continue

        processor_start = time.perf_counter()

        # Decode frame
        img = decode_frame(frame_bytes)

        # Run inference
        output, inference_time = inference_engine.run_inference(img)
        print(f"  ▸ Optimized Inference: {inference_time:.1f} ms")

        # Process detections
        detections = DetectionProcessor.process_detections(output)

        # Update tracking
        tracks = tracking_manager.update_tracks(detections, img)
        track_info = TrackingManager.extract_track_info(tracks)

        # Extract crops only for objects needing OCR
        sort_start = time.perf_counter()
        crops, crop_metadata, track_ids_needing_ocr = CropExtractor.extract_crops(
            img, track_info, object_cache
        )

        # Process OCR only for new objects
        if crops:
            print(f"  ▸ Processing OCR for {len(crops)} new objects")
            ocr_results = ocr_processor.process_crops(crops)

            # Update cache with OCR results
            for (x1, y1, x2, y2, track_id, class_name, score), ocr_lines in zip(crop_metadata, ocr_results):
                filtered_ocr = OCRProcessor.filter_ocr_results(ocr_lines)
                object_cache.update_object(track_id, (x1, y1, x2, y2), frame_id, class_name, filtered_ocr)
        else:
            print("  ▸ No new objects need OCR processing")

        # Update cache for all tracked objects (even those without new OCR)
        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            if track_id not in track_ids_needing_ocr:
                object_cache.update_object(track_id, (x1, y1, x2, y2), frame_id, class_name, [])

        # Cleanup old tracks
        object_cache.cleanup_old_tracks(frame_id)

        sort_end = time.perf_counter()
        print(f"  ▸ SORT & Cache: {(sort_end - sort_start) * 1000:.1f} ms")

        # Create panel rows
        panel_rows = ResultAssembler.create_panel_rows(track_info, object_cache)

        # Draw results
        draw_start = time.perf_counter()
        labels, boxes, scores = output["labels"], output["boxes"], output["scores"]
        pil_img = draw([Image.fromarray(img)], labels, boxes, scores, DETECTION_CONFIDENCE_THRESHOLD)
        draw_end = time.perf_counter()
        print(f"  ▸ Draw: {(draw_end - draw_start) * 1000:.1f} ms")

        # Assemble final output
        data = {
            "frame_id": frame_id,
            "img": img,
            "pil_img": pil_img[0],
            "panel_rows": panel_rows,
        }

        processor_end = time.perf_counter()
        print(f"Total Processing: {(processor_end - processor_start) * 1000:.1f} ms")

        output_queue.put(data)
        frame_id += 1