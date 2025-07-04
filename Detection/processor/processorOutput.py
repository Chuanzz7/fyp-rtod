import queue
import time
from multiprocessing import Queue
from typing import List, Dict

import cv2

from Detection.helper import videoHelper
from Detection.helper.apiHelper import APIManager
from Detection.helper.metricHelper import update_metric

api_called_ids = set()
api_id_last_seen = dict()
api_results = dict()  # {track_id: {"code": str, "confidence": float, "timestamp": float}}


def process_output_main(input_queue: Queue, mjpeg_frame_queue: Queue, shared_config, shared_metrics):
    """
    Main processing loop for handling detection outputs, calling APIs,
    and generating the final video stream frame.
    """
    # Instantiate the APIManager ONCE before the loop
    api_manager = APIManager(api_id_max_age=150)  # Increased age to ~5 seconds at 30fps

    # For FPS calculation
    last_fps_time = time.perf_counter()
    frame_counter = 0

    try:
        while True:
            try:
                # This is the data structure you provided
                data = input_queue.get(timeout=1)
            except queue.Empty:
                continue

            stage_start_time = time.perf_counter()
            frame_id = data["frame_id"]
            assigned_regions = list(shared_config.get('assigned_regions', []))

            # 1. Placement Check (CPU-bound, fast)
            draw_start_time = time.perf_counter()
            checked_panel_rows = check_wrong_placement(
                data["panel_rows"], assigned_regions, iou_threshold=0.1
            )
            update_metric(shared_metrics, "output_placement_time_ms", (time.perf_counter() - draw_start_time) * 1000)

            # 2. Asynchronous API Calls (Non-blocking)
            api_start = time.perf_counter()
            # This is now very fast. It just queues tasks for the background thread.
            # We use the actual frame_id from the data packet.
            api_manager.process_and_call_api(checked_panel_rows, frame_id)

            # 3. Enrich panel data with the *current* API results
            current_api_results = api_manager.get_api_results()
            enriched_panel_rows = enrich_rows_with_api_results(checked_panel_rows, current_api_results)
            update_metric(shared_metrics, "output_api_time_ms", (time.perf_counter() - api_start) * 1000)

            # 4. Draw, build panel, and composite frame
            draw_start_time = time.perf_counter()
            np_img_with_boxes = videoHelper.draw_assigned_regions_on_frame(
                data["np_img"], assigned_regions
            )
            panel = videoHelper.build_detection_panel(enriched_panel_rows, data["np_img"].shape[0])
            composite = videoHelper.side_by_side(np_img_with_boxes, panel)
            update_metric(shared_metrics, "output_draw_time_ms", (time.perf_counter() - draw_start_time) * 1000)

            # 5. Encode to JPEG
            encode_start_time = time.perf_counter()
            success, encoded_image = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue
            jpeg_bytes = encoded_image.tobytes()
            update_metric(shared_metrics, "output_encode_time_ms", (time.perf_counter() - encode_start_time) * 1000)

            # 6. Send frame to MJPEG server
            try:
                mjpeg_frame_queue.put_nowait(jpeg_bytes)
            except queue.Full:
                pass  # Drop frames if the consumer is backed up

            # 7. Calculate and update FPS
            frame_counter += 1
            now = time.perf_counter()
            elapsed = now - last_fps_time
            update_metric(shared_metrics, "output_total_processing_time_ms", (now - stage_start_time) * 1000)
            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                update_metric(shared_metrics, "output_fps", fps)
                frame_counter = 0
                last_fps_time = now


    except (KeyboardInterrupt, SystemExit):
        print("Process output main shutting down...")
    finally:
        # CRITICAL: Ensure the API manager is shut down gracefully
        api_manager.shutdown()


def enrich_rows_with_api_results(panel_rows: List[Dict], api_results: Dict) -> List[Dict]:
    for row in panel_rows:
        track_id = int(row["object_id"])
        if track_id in api_results:
            row["api_result"] = api_results[track_id]
    return panel_rows


def check_wrong_placement(panel_rows, assigned_regions, iou_threshold=0.1):
    """
    For each detection, flag as:
      - 'Correct Region' if matches assigned label in that region
      - 'Wrong Region' if label/ocr does NOT match region's assignment
      - 'Unassigned' if not in any region
    Returns: list of dicts (detection + status)
    """
    results = []

    for row in panel_rows:
        det_bbox = row["box"]
        det_label = row["class_name"]

        status = "Unassigned"
        matched_region_label = None

        # Check against each assigned region
        for reg in assigned_regions:
            reg_bbox = reg["bbox"]
            iou = compute_iou(det_bbox, reg_bbox)

            if iou > iou_threshold:
                matched_region_label = reg["label"]

                if det_label.lower() == reg["label"].lower():
                    status = "Correct Region"
                else:
                    status = f"Wrong Region ({reg['label']})"

        # Create result with all original data plus status
        result = {
            **row,  # Keep all original keys for panel code
            "region_status": status,
            "matched_region": matched_region_label,
        }
        results.append(result)

    return results


def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Assumes bbox format: [x1, y1, x2, y2] or [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """

    # Handle different bbox formats
    def normalize_bbox(bbox):
        if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
            # Format: [x1, y1, x2, y2]
            return bbox
        elif len(bbox) == 4 and all(isinstance(point, (list, tuple)) for point in bbox):
            # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] - convert to [x1,y1,x2,y2]
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
        else:
            raise ValueError(f"Unsupported bbox format: {bbox}")

    try:
        box1 = normalize_bbox(bbox1)
        box2 = normalize_bbox(bbox2)

        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # No intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union
        union = area1 + area2 - intersection

        # Avoid division by zero
        if union == 0:
            return 0.0

        return intersection / union

    except Exception as e:
        print(f"Error computing IoU for {bbox1} and {bbox2}: {e}")
        return 0.0
