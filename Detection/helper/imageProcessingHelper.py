import base64
import time
from collections import deque
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

# In your constants section or at the start of processor_tensor_main
VIDEO_FPS = 20  # Assumed FPS of your input video stream
OCR_INTERVAL_SECONDS = 2.0
OCR_INTERVAL_FRAMES = int(VIDEO_FPS * OCR_INTERVAL_SECONDS)
MIN_FRAMES_FOR_OCR = 5  # The number of frames an object must be stable before we start OCR
MAX_OCR_HISTORY_PER_OBJECT = 20


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

    def __init__(self, warmup: bool = True):
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

            # Advanced settings for curved/distorted text (bottles)
            use_space_char=True,  # Preserve spaces in text
            drop_score=0.3,  # Lower score threshold for keeping results (default: 0.5)
        )

        self.is_warmed_up = False

        # Perform warm-up by default
        if warmup:
            self.warmup()

    def warmup(self, num_warmup_runs: int = 3):
        """
        Warm up the OCR models to avoid cold start delays.

        Args:
            num_warmup_runs: Number of warm-up inference runs
        """
        print("🔥 Warming up OCR models...")
        warmup_start = time.perf_counter()

        # Create different sized dummy images to warm up all model components
        dummy_images = [
            self._create_dummy_image(320, 240),  # Small image
            self._create_dummy_image(640, 480),  # Medium image
            self._create_dummy_image(800, 600),  # Large image
        ]

        for i in range(num_warmup_runs):
            for j, dummy_img in enumerate(dummy_images):
                try:
                    # Warm up with different configurations
                    _ = self.ocr.ocr(
                        dummy_img,
                        det=True,
                        cls=True,
                    )
                    print(f"  ▸ Warmup run {i + 1}/{num_warmup_runs}, image {j + 1}/{len(dummy_images)} completed")
                except Exception as e:
                    print(f"  ⚠️  Warmup warning: {e}")

        warmup_end = time.perf_counter()
        warmup_time = (warmup_end - warmup_start) * 1000
        print(f"✅ OCR models warmed up in {warmup_time:.1f} ms")
        self.is_warmed_up = True

    def _create_dummy_image(self, width: int, height: int) -> np.ndarray:
        """Create a dummy image with some text for warm-up"""
        # Create white background
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Add some dummy text using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0)  # Black text
        thickness = 2

        # Add text at different positions
        texts = ["PRODUCT", "LABEL", "TEXT", "123"]
        y_positions = [height // 4, height // 2, 3 * height // 4, height - 50]

        for text, y_pos in zip(texts, y_positions):
            if y_pos > 0 and y_pos < height:
                cv2.putText(img, text, (50, y_pos), font, font_scale, color, thickness)

        return img

    def ensure_warmed_up(self):
        """Ensure the model is warmed up before processing"""
        if not self.is_warmed_up:
            print("⚠️  Model not warmed up, warming up now...")
            self.warmup()


    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses a crop for better OCR performance by automatically handling
        both light-on-dark and dark-on-light text.
        """
        if crop is None or crop.size == 0:
            return None

        # 1. Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        mean_intensity = np.mean(gray)

        if mean_intensity < 127:
            gray = cv2.bitwise_not(gray)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def process_single_crop(self, crop: np.ndarray) -> List[Dict]:
        """Process single image crop with OCR - optimized for product text"""
        if crop.size == 0:
            return []

        # Ensure model is warmed up
        self.ensure_warmed_up()

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

    def process_crops_dict(self, crops_dict: Dict[int, np.ndarray]) -> Dict[int, List]:
        """
        Processes a dictionary of crops keyed by track_id and returns a
        dictionary of OCR results with the same track_id keys.
        """
        if not crops_dict:
            return {}

        self.ensure_warmed_up()

        # Prepare lists for batch processing while keeping track of original IDs
        track_ids = list(crops_dict.keys())
        crops_list = [crops_dict[tid] for tid in track_ids]

        # Use the batch processing capability of PaddleOCR if available,
        # otherwise iterate. Iteration is safer and shown here.
        results_dict = {}
        for track_id, crop in crops_dict.items():
            # processed_crop = self._preprocess_crop(crop)
            # The result from ocr() is a list of lines for this single crop
            ocr_lines = self.ocr.ocr(crop, det=True, cls=True)
            results_dict[track_id] = ocr_lines

        return results_dict


class ObjectCache:
    """
    Manages object tracking cache and intelligently aggregates OCR results over time.
    """

    def __init__(self, ocr_interval_frames: int = OCR_INTERVAL_FRAMES):
        self.cache: Dict[int, Dict[str, Any]] = {}
        self.ocr_interval_frames = ocr_interval_frames
        print(f"✅ ObjectCache initialized. OCR will be triggered for stable tracks every {ocr_interval_frames} frames.")

    def update_object(self, track_id: int, box: Tuple[int, int, int, int],
                      frame_id: int, class_name: str, new_ocr_results: List[Dict]) -> None:
        """Update object in cache with new tracking and OCR data, with history limits."""
        x1, y1, x2, y2 = box

        if track_id not in self.cache:
            self.cache[track_id] = {
                "history": deque(maxlen=MAX_OCR_HISTORY_PER_OBJECT),
                "first_seen": frame_id,
                "frames_tracked": 0,
                "ocr_readings_history": deque(maxlen=MAX_OCR_HISTORY_PER_OBJECT),
                "last_ocr_frame": -1,
                "last_seen": frame_id,
                "class_name": class_name,
            }

        # Update tracking info (you could also limit this history if needed, using the same technique)
        self.cache[track_id]["history"].append((x1, y1, x2, y2, frame_id))
        self.cache[track_id]["last_seen"] = frame_id
        self.cache[track_id]["class_name"] = class_name
        self.cache[track_id]["frames_tracked"] += 1

        # If new OCR results were provided, add them and update the last OCR frame
        if new_ocr_results:
            self.cache[track_id]["ocr_readings_history"].extend(new_ocr_results)
            self.cache[track_id]["last_ocr_frame"] = frame_id

    def needs_ocr(self, track_id: int, current_frame_id: int, min_frames: int = MIN_FRAMES_FOR_OCR) -> bool:
        """
        Check if OCR processing is needed for this track.
        - The object must be tracked for a minimum number of frames.
        - Enough time (frames) must have passed since the last OCR.
        """
        if track_id not in self.cache:
            return False  # Should not happen if called on a tracked object

        tracked_obj = self.cache[track_id]

        is_stable = tracked_obj["frames_tracked"] >= min_frames
        is_time_for_next_ocr = (current_frame_id - tracked_obj["last_ocr_frame"]) >= self.ocr_interval_frames

        return is_stable and is_time_for_next_ocr

    def get_aggregated_ocr_results(self, track_id: int) -> List[Dict]:
        """
        Get aggregated and ranked OCR results for a track.
        This function counts occurrences of each text and returns the most common ones first.
        """
        if track_id not in self.cache or not self.cache[track_id]["ocr_readings_history"]:
            return []

        text_aggregator: Dict[str, Dict[str, Any]] = {}
        for reading in self.cache[track_id]["ocr_readings_history"]:
            text = reading["text"]
            confidence = reading["confidence"]
            if text not in text_aggregator:
                text_aggregator[text] = {"count": 0, "confidence_sum": 0.0}
            text_aggregator[text]["count"] += 1
            text_aggregator[text]["confidence_sum"] += confidence

        # Create a list of results with aggregated data
        aggregated_list = []
        for text, data in text_aggregator.items():
            aggregated_list.append({
                "text": text,
                "count": data["count"],
                "avg_confidence": round(data["confidence_sum"] / data["count"], 3)
            })

        # Sort the list by count (most frequent first), then by confidence
        return sorted(aggregated_list, key=lambda x: (x["count"], x["avg_confidence"]), reverse=True)

    def get_best_ocr_texts(self, track_id: int, min_count: int = 2, min_confidence: float = 0.65) -> List[str]:
        """
        Filters the aggregated results to return a clean list of the most reliable text strings.
        This is the perfect input for fuzzy searching.

        Args:
            track_id: The ID of the object to get texts for.
            min_count: The minimum number of times a text must be seen to be included.
            min_confidence: The minimum average confidence score for a text to be included.

        Returns:
            A list of high-quality OCR text strings.
        """
        aggregated_results = self.get_aggregated_ocr_results(track_id)
        if not aggregated_results:
            return []

        best_texts = []
        for result in aggregated_results:
            # Apply filters to select only the most reliable text
            if result['count'] >= min_count and result['avg_confidence'] >= min_confidence:
                best_texts.append(result['text'])

        # As a final refinement, you could also just return the top N results
        # For example, to return only the top 3 most frequent & confident texts:
        # return [result['text'] for result in aggregated_results[:3]]

        return best_texts

    def cleanup_old_tracks(self, current_frame: int,
                           max_disappear: int = MAX_TRACK_DISAPPEAR_FRAMES) -> None:
        """Remove old tracks from cache."""
        remove_ids = [
            oid for oid, data in self.cache.items()
            if current_frame - data["last_seen"] > max_disappear
        ]
        for oid in remove_ids:
            # print(f"  ▸ Removing expired track {oid} from cache.")
            del self.cache[oid]


class CropExtractor:
    """Extracts image crops from tracked objects"""

    @staticmethod
    def extract_crops_as_dict(img: np.ndarray, track_info: List[Tuple], object_cache: ObjectCache,
                              current_frame_id: int, min_frames: int = 15) -> Dict[int, np.ndarray]:
        """
        Extracts  crops for objects needing OCR and returns them in a
        dictionary keyed by their track_id.
        """
        crops_to_process = {}
        img_h, img_w, _ = img.shape  # Get image boundaries

        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            if object_cache.needs_ocr(track_id, current_frame_id, min_frames=min_frames):
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops_to_process[track_id] = crop
        return crops_to_process


class ResultAssembler:
    """Assembles final results for output"""

    @staticmethod
    def create_panel_rows(track_info: List[Tuple], object_cache: ObjectCache) -> List[Dict]:
        """Create panel rows with all tracking and aggregated OCR information"""
        panel_rows = []

        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            # Get the full aggregated data for detailed analysis (optional)
            aggregated_results = object_cache.get_aggregated_ocr_results(track_id)

            # Get the clean, filtered list of strings for fuzzysearch
            best_texts_for_search = object_cache.get_best_ocr_texts(track_id)

            row = {
                "object_id": track_id,
                "class_name": class_name,
                "confidence": round(float(score or 1.0), 3),
                "box": [x1, y1, x2, y2],
                # Use the new aggregation method
                "ocr_results": aggregated_results,  # The full data
                "best_ocr_texts": best_texts_for_search  # The clean list for fuzzysearch
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
