import queue
import queue
import time
from multiprocessing import Queue

from PIL import Image

from DFINE.tools.inference.trt_inf import draw
from Detection.helper import drawHelper
from Detection.helper.imageProcessingHelper import GPUBufferManager, InferenceEngine, TrackingManager, OCRProcessor, \
    ObjectCache, decode_frame, DetectionProcessor, CropExtractor, ResultAssembler


def processor_tensor_main(frame_input_queue: Queue, output_queue: Queue, shared_metrics):
    """Main processing function with refactored components - ORIGINAL QUEUE-BASED PIPELINE"""
    # Initialize components
    GPUBufferManager.initialize_buffers()
    inference_engine = InferenceEngine()
    tracking_manager = TrackingManager()
    ocr_processor = OCRProcessor(warmup=True)
    object_cache = ObjectCache()
    N = 120  # or any number of frames you want to keep

    print("All components initialized, waiting for frames...")

    frame_id = 0
    while True:
        try:
            frame_bytes = frame_input_queue.get(timeout=1)
        except queue.Empty:
            object_cache.cleanup_old_tracks(frame_id, 1)
            continue

        processor_start = time.perf_counter()

        # Decode frame
        decode_start = time.perf_counter()
        img = decode_frame(frame_bytes)
        decode_end = time.perf_counter()

        # Run inference
        output, inference_time = inference_engine.run_inference(img)

        # Process detections
        detections = DetectionProcessor.process_detections(output)

        # Update tracking
        tracks = tracking_manager.update_tracks(detections, img)
        track_info = TrackingManager.extract_track_info(tracks)

        # Extract crops only for objects needing OCR
        sort_start = time.perf_counter()
        crops, crop_metadata, track_ids_needing_ocr = CropExtractor.extract_crops(
            img, track_info, object_cache, min_frames=5
        )

        track_info_map = {ti[4]: ti for ti in track_info}  # ti[4] is the track_id

        crops_to_process = CropExtractor.extract_crops_as_dict(
            img, track_info, object_cache, min_frames=5
        )

        # 2. Process OCR only if there are crops
        if crops_to_process:
            ocr_start = time.perf_counter()
            # The result is a dictionary {track_id: ocr_result}
            ocr_results_dict = ocr_processor.process_crops_dict(crops_to_process)
            ocr_end = time.perf_counter()

            shared_metrics.setdefault("ocr_time_ms", []).append((ocr_end - ocr_start) * 1000)
            shared_metrics["ocr_time_ms"][:] = shared_metrics["ocr_time_ms"][-N:]

            # 3. Update cache using the correctly-keyed OCR results
            for track_id, ocr_lines in ocr_results_dict.items():
                # Check if the track still exists in the current frame
                if track_id in track_info_map:
                    # Get the most recent bounding box for this track_id
                    x1, y1, x2, y2, _, class_name, _ = track_info_map[track_id]
                    filtered_ocr = OCRProcessor.filter_ocr_results(ocr_lines)
                    # Update cache with OCR. The box update is for the current frame.
                    object_cache.update_object(track_id, (x1, y1, x2, y2), frame_id, class_name, filtered_ocr)

        # Update cache for all tracked objects (even those without new OCR)
        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            if track_id not in track_ids_needing_ocr:
                object_cache.update_object(track_id, (x1, y1, x2, y2), frame_id, class_name, [])

        # Cleanup old tracks
        object_cache.cleanup_old_tracks(frame_id)
        sort_end = time.perf_counter()

        # Create panel rows
        panel_rows = ResultAssembler.create_panel_rows(track_info, object_cache)

        # Draw results
        draw_start = time.perf_counter()
        labels, boxes, scores = output["labels"], output["boxes"], output["scores"]
        np_img = drawHelper.draw(img, labels, boxes, scores, thrh=0.8)
        draw_end = time.perf_counter()

        # Assemble final output
        data = {
            "frame_id": frame_id,
            "img": img,
            "np_img": np_img,
            "panel_rows": panel_rows,
        }

        processor_end = time.perf_counter()
        output_queue.put(data)
        frame_id += 1

        # Decode
        shared_metrics.setdefault("decode_time_ms", []).append((decode_end - decode_start) * 1000)
        shared_metrics["decode_time_ms"][:] = shared_metrics["decode_time_ms"][-N:]

        # D-Fine Inference
        shared_metrics.setdefault("dfine_inference_time_ms", []).append(inference_time)
        shared_metrics["dfine_inference_time_ms"][:] = shared_metrics["dfine_inference_time_ms"][-N:]

        # SORT & Cache
        shared_metrics.setdefault("sort_and_cache_time_ms", []).append((sort_end - sort_start) * 1000)
        shared_metrics["sort_and_cache_time_ms"][:] = shared_metrics["sort_and_cache_time_ms"][-N:]

        # Draw time
        shared_metrics.setdefault("draw_time_ms", []).append((draw_end - draw_start) * 1000)
        shared_metrics["draw_time_ms"][:] = shared_metrics["draw_time_ms"][-N:]

        # Total processing time
        shared_metrics.setdefault("total_processing_time_ms", []).append((processor_end - processor_start) * 1000)
        shared_metrics["total_processing_time_ms"][:] = shared_metrics["total_processing_time_ms"][-N:]
