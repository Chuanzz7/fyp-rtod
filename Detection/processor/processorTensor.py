import queue
import time
from multiprocessing import Queue

from Detection.helper import drawHelper
from Detection.helper.imageProcessingHelper import GPUBufferManager, InferenceEngine, TrackingManager, OCRProcessor, \
    ObjectCache, decode_frame, DetectionProcessor, CropExtractor, ResultAssembler
from Detection.helper.metricHelper import update_metric


def processor_tensor_main(frame_input_queue: Queue, output_queue: Queue, shared_metrics):
    # Initialize components
    GPUBufferManager.initialize_buffers()
    inference_engine = InferenceEngine()
    tracking_manager = TrackingManager()
    ocr_processor = OCRProcessor(warmup=True)
    object_cache = ObjectCache(ocr_interval_frames=30)

    print("All components initialized, waiting for frames...")

    frame_id = 0
    while True:
        try:
            frame_bytes = frame_input_queue.get(timeout=1)
        except queue.Empty:
            object_cache.cleanup_old_tracks(frame_id, 1)
            continue

        processor_start = time.perf_counter()

        decode_start = time.perf_counter()
        img = decode_frame(frame_bytes)
        decode_end = time.perf_counter()
        output, inference_time = inference_engine.run_inference(img)
        detections = DetectionProcessor.process_detections(output)
        tracks = tracking_manager.update_tracks(detections, img)
        track_info = TrackingManager.extract_track_info(tracks)

        # --- UPDATE: Update cache for all currently tracked objects FIRST ---
        # This ensures 'frames_tracked' is up-to-date before checking 'needs_ocr'
        for x1, y1, x2, y2, track_id, class_name, score in track_info:
            object_cache.update_object(track_id, (x1, y1, x2, y2), frame_id, class_name,
                                       [])  # Pass empty OCR results for now

        # --- UPDATE: Pass current_frame_id to the crop extractor ---
        crops_to_process = CropExtractor.extract_crops_as_dict(
            img, track_info, object_cache, current_frame_id=frame_id, min_frames=5
        )

        # Process OCR only if there are crops that meet the new criteria
        if crops_to_process:
            ocr_start = time.perf_counter()
            ocr_results_dict = ocr_processor.process_crops_dict(crops_to_process)
            ocr_end = time.perf_counter()
            update_metric(shared_metrics, "ocr_time_ms", (ocr_end - ocr_start) * 1000)

            # --- UPDATE: Update cache AGAIN, this time with the new OCR results ---
            for track_id, ocr_lines in ocr_results_dict.items():
                # We don't need to check if the track exists, because it must exist to have been selected for OCR
                current_box = object_cache.cache[track_id]["history"][-1][:4]  # Get the most recent box
                current_class_name = object_cache.cache[track_id]["class_name"]
                filtered_ocr = OCRProcessor.filter_ocr_results(ocr_lines)

                # This call will now APPEND the new OCR results and update the last_ocr_frame
                object_cache.update_object(track_id, current_box, frame_id, current_class_name, filtered_ocr)

        # Cleanup old tracks
        object_cache.cleanup_old_tracks(frame_id)

        # Create panel rows with aggregated results
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
        update_metric(shared_metrics, "decode_time_ms", (decode_end - decode_start) * 1000)

        # D-Fine Inference
        update_metric(shared_metrics, "dfine_inference_time_ms", inference_time)

        # SORT & Cache
        # update_metric(shared_metrics, "sort_and_cache_time_ms", (sort_end - sort_start) * 1000)

        # Draw time
        update_metric(shared_metrics, "draw_time_ms", (draw_end - draw_start) * 1000)

        # Total processing time
        update_metric(shared_metrics, "total_processing_time_ms", (processor_end - processor_start) * 1000)
