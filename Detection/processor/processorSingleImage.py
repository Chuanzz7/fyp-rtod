import time
from typing import Dict, Any, Union

import numpy as np
from PIL import Image
from fastapi import HTTPException

from Detection.helper.imageProcessingHelper import GPUBufferManager, InferenceEngine, OCRProcessor, DetectionProcessor, \
    ResultAssembler

DETECTION_CONFIDENCE_THRESHOLD = 0.8
OCR_CONFIDENCE_THRESHOLD = 0.8
_input_tensor = None


class SingleImageProcessor:
    """Handles single image processing for API calls"""

    def __init__(self):
        """Initialize the processor with required components"""
        # Initialize GPU buffers if not already done
        global _input_tensor
        if _input_tensor is None:
            GPUBufferManager.initialize_buffers()

        # Initialize components
        self.inference_engine = InferenceEngine()
        self.ocr_processor = OCRProcessor(warmup=True)
        self.result_assembler = ResultAssembler()

        print("Single image processor initialized")

    def process_image(self,
                      image_input: Union[str, bytes, np.ndarray, Image.Image],
                      include_ocr: bool = True,
                      detection_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
                      ocr_threshold: float = OCR_CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
        """
        Process a single image and return detection and OCR results

        Args:
            image_input: Input image (file path, base64 string, bytes, numpy array, or PIL Image)
            include_ocr: Whether to perform OCR on detected objects
            detection_threshold: Confidence threshold for object detection
            ocr_threshold: Confidence threshold for OCR

        Returns:
            Dictionary containing detection results and OCR information
        """
        processor_start = time.perf_counter()

        # Convert input to numpy array
        img = self.result_assembler._prepare_image(image_input)

        # Run inference
        output, inference_time = self.inference_engine.run_inference(img)
        print(f"  ▸ Inference: {inference_time:.1f} ms")

        # Process detections for API format
        detections = DetectionProcessor.process_detections_for_api(output, detection_threshold)

        # Add OCR results if requested
        if include_ocr:
            ocr_start = time.perf_counter()
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                crop = img[y1:y2, x1:x2]

                if crop.size > 0:
                    ocr_results = self.ocr_processor.process_single_crop(crop)
                    detection["ocr_results"] = ocr_results
                else:
                    detection["ocr_results"] = []

            ocr_end = time.perf_counter()
            print(f"  ▸ Total OCR: {(ocr_end - ocr_start) * 1000:.1f} ms")
        else:
            for detection in detections:
                detection["ocr_results"] = []

        processor_end = time.perf_counter()
        total_time = (processor_end - processor_start) * 1000

        if len(detections) != 1:
            # Throw HTTP error or return error dict (FastAPI example below)
            raise HTTPException(status_code=400, detail=f"Expected exactly 1 detection, found {len(detections)}.")

        # Combine OCR texts from the single detection
        ocr_results = detections[0].get("ocr_results", [])
        ocr_texts = []
        for result in ocr_results:
            # Adapt to your OCR output format; adjust as needed
            if isinstance(result, dict):
                text = result.get("text", "")
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                text = result[0]
            else:
                text = str(result)
            if text:
                ocr_texts.append(text)
        combined_ocr_text = " ".join(ocr_texts)

        # Prepare response for a single detection
        response = {
            "status": "success",
            "processing_time_ms": round(total_time, 1),
            "inference_time_ms": round(inference_time, 1),
            "image_dimensions": {
                "width": img.shape[1],
                "height": img.shape[0]
            },
            "detections_count": 1,
            "detections": [detections[0]],  # Only the single detection
            "ocrTexts": combined_ocr_text,  # One line field with all OCR text
            "settings": {
                "detection_threshold": detection_threshold,
                "ocr_threshold": ocr_threshold,
                "ocr_enabled": include_ocr
            }
        }
        print(f"Total processing: {total_time:.1f} ms")
        return response
