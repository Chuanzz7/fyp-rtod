import queue
import queue
import time
from multiprocessing import Queue

from PIL import Image

from DFINE.tools.inference.trt_inf import draw
from Detection.helper.imageProcessingHelper import GPUBufferManager, InferenceEngine, TrackingManager, OCRProcessor, \
    ObjectCache, decode_frame, DetectionProcessor, CropExtractor, ResultAssembler


def processor_tensor_main(frame_input_queue: Queue, output_queue: Queue):
    """Main processing function with refactored components - ORIGINAL QUEUE-BASED PIPELINE"""
    # Initialize components
    GPUBufferManager.initialize_buffers()
    inference_engine = InferenceEngine()
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
        pil_img = draw([Image.fromarray(img)], labels, boxes, scores, 0.8)
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
