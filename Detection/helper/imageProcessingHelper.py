import base64
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from paddleocr import PaddleOCR

from DFINE.tools.inference.trt_inf import TRTInference
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

    def __init__(self, model_path=TENSOR_MODEL):
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

    @staticmethod
    def process_detections_for_api(output: Dict[str, torch.Tensor],
                                   confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD) -> List[Dict]:
        """Convert model output to API-friendly format"""
        labels, boxes, scores = output["labels"], output["boxes"], output["scores"]
        detections = []

        for i in range(boxes.shape[1]):
            score = scores[0, i].item()
            if score < confidence_threshold:
                continue

            x1, y1, x2, y2 = [int(boxes[0, i, j].item()) for j in range(4)]
            label = int(labels[0, i].item())
            class_name = COCO_CLASSES[label][1] if 0 <= label < len(COCO_CLASSES) else "unknown"

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": score,
                "class_name": class_name,
                "class_id": label
            })

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
    """Optimized OCR processor for product text detection (bottles, packages, etc.)"""

    def __init__(self):
        self.ocr = PaddleOCR(
            # Core model settings
            det_model_dir=None,  # Use default v5 detection model
            rec_model_dir=None,  # Use default v5 recognition model
            cls_model_dir=None,  # Use default v5 classification model

            # Language and processing
            lang="en",  # Set to specific language for better accuracy
            use_angle_cls=True,  # Essential for rotated text on curved bottles

            # Performance settings
            use_gpu=True,  # GPU acceleration
            enable_mkldnn=True,  # CPU optimization when GPU unavailable
            cpu_threads=4,  # Optimize for your CPU

            # Detection parameters - crucial for small/curved text
            det_algorithm="DB",  # Only supported algorithm in v5
            det_limit_type="max",  # Use 'max' for high-res product images
            det_limit_side_len=1536,  # Higher resolution for small text (default: 960)
            det_db_thresh=0.3,  # Lower threshold for faint text (default: 0.3)
            det_db_box_thresh=0.5,  # Box filtering threshold (default: 0.6)
            det_db_unclip_ratio=1.6,  # Expand detection boxes slightly (default: 1.5)
            det_db_score_mode="fast",  # Use 'fast' for better compatibility

            # Recognition parameters - use default algorithms
            # rec_algorithm is auto-selected in v5, don't specify explicitly
            rec_image_shape="3, 48, 320",  # Optimized for product text
            rec_batch_num=6,  # Batch processing

            # Quality and preprocessing
            use_dilation=True,  # Helps with thin text
            det_east_score_thresh=0.8,  # For EAST algorithm if used
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,

            # Output settings
            show_log=False,  # Set to True for debugging
            save_crop_res=False,  # Set to True if you want to save cropped regions
            crop_res_save_dir="./output",  # Directory for saved crops

            # Advanced settings for curved/distorted text (bottles)
            use_space_char=True,  # Preserve spaces in text
            drop_score=0.3,  # Lower score threshold for keeping results (default: 0.5)
        )

    def process_crops(self, crops: List[np.ndarray]) -> List:
        """Process multiple image crops with OCR - optimized for products"""
        if not crops:
            return []


        # Process with optimized parameters for product text
        results = []
        for crop in crops:
            # Preprocess crop for better OCR
            processed_crop = self._preprocess_crop(crop)
            result = self.ocr.ocr(
                processed_crop,
                det=True,
                cls=True,
                # Additional parameters for better product text detection
            )
            results.append(result)

        return results

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess image crop for better OCR on products"""
        import cv2

        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Enhance contrast for faint text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Convert back to BGR for PaddleOCR
        if len(crop.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def process_single_crop(self, crop: np.ndarray) -> List[Dict]:
        """Process single image crop with OCR - optimized for product text"""
        if crop.size == 0:
            return []

        t_start = time.perf_counter()

        # Preprocess for better results
        processed_crop = self._preprocess_crop(crop)

        result = self.ocr.ocr(
            processed_crop,
            det=True,
            cls=True,
        )

        t_end = time.perf_counter()
        print(f"  ▸ OCR processing: {(t_end - t_start) * 1000:.1f} ms")
        return self.filter_ocr_results(result)

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
                                "text": txt,
                                "confidence": round(conf, 3)
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
                "frames_tracked": 1,
                "ocr_results": [],
                "last_seen": frame_id,
                "class_name": class_name,
                "ocr_processed": False,  # Flag to track if OCR has been done
                "api_called": False,
            }
        else:
            # Increment frames_tracked if seen in consecutive frames
            self.cache[track_id]["frames_tracked"] += 1

        # Update tracking info
        self.cache[track_id]["history"].append((x1, y1, x2, y2, frame_id))
        self.cache[track_id]["last_seen"] = frame_id
        self.cache[track_id]["class_name"] = class_name

        # Only update OCR if we haven't processed it yet and we have new results
        if not self.cache[track_id]["ocr_processed"] and ocr_results:
            self.cache[track_id]["ocr_results"] = ocr_results
            self.cache[track_id]["ocr_processed"] = True
            print(f"  ▸ OCR saved for track {track_id}: {len(ocr_results)} results")

    def needs_ocr(self, track_id: int, min_frames: int = 2) -> bool:
        """Check if OCR processing is needed for this track, and has survived min_frames"""
        if track_id not in self.cache:
            return False  # Don't OCR unknown tracks!
        tracked = self.cache[track_id]
        return (
                not tracked["ocr_processed"]
                and tracked["frames_tracked"] >= min_frames
        )

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
    def extract_crops(img: np.ndarray, track_info: List[Tuple], object_cache: ObjectCache, min_frames: int = 5):
        crops = []
        crop_metadata = []
        track_ids_needing_ocr = []

        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            if object_cache.needs_ocr(track_id, min_frames=min_frames):
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
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

    def _prepare_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Convert various image input formats to numpy array"""

        if isinstance(image_input, str):
            # Handle file path or base64 string
            if image_input.startswith('data:image') or len(image_input) > 500:
                # Assume base64 string
                if image_input.startswith('data:image'):
                    # Remove data URL prefix
                    image_input = image_input.split(',')[1]

                image_bytes = base64.b64decode(image_input)
                img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("Failed to decode base64 image")

                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Assume file path
                img = cv2.imread(image_input)
                if img is None:
                    raise ValueError(f"Failed to load image from path: {image_input}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, bytes):
            # Handle raw bytes
            img_array = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image bytes")

            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, np.ndarray):
            # Handle numpy array
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume RGB format
                return image_input
            else:
                raise ValueError("Numpy array must be 3-channel RGB image")

        elif isinstance(image_input, Image.Image):
            # Handle PIL Image
            if image_input.mode != 'RGB':
                image_input = image_input.convert('RGB')
            return np.array(image_input)

        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")


def decode_frame(frame_bytes: bytes) -> np.ndarray:
    """Decode JPEG frame bytes to OpenCV image"""
    np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_raw is None:
        raise ValueError("Failed to decode JPEG image")

    return cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
