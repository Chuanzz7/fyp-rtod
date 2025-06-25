import asyncio
import queue
import time
from multiprocessing import Queue

import cv2
import numpy as np

from Detection.helper import videoHelper

api_called_ids = set()
api_id_last_seen = dict()
api_results = dict()  # {track_id: {"code": str, "confidence": float, "timestamp": float}}


def process_output_main(input_queue: Queue, mjpeg_frame_queue: Queue, shared_config, shared_metrics):
    last_time = time.perf_counter()
    frame_count = 0
    fps = 0
    N = 120  # keep last 120 samples for dashboard

    while True:
        try:
            data = input_queue.get(timeout=1)
        except queue.Empty:
            api_called_ids.clear()
            api_id_last_seen.clear()
            api_results.clear()
            continue

        stage_start = time.perf_counter()

        # ============ 1. Build panel rows (placement check)
        panel_start = time.perf_counter()
        checked_panel_rows = check_wrong_placement(
            data["panel_rows"], list(shared_config.get('assigned_regions', [])),
            iou_threshold=0.1
        )
        panel_end = time.perf_counter()

        # ============ 2. Async API calls (timing optional)
        api_start = time.perf_counter()
        asyncio.run(
            call_product_api_from_panel(checked_panel_rows, frame_count, 100))
        api_end = time.perf_counter()

        # ============ 3. Panel result enrichment
        enrich_start = time.perf_counter()
        enriched_panel_rows = add_api_results_to_panel(checked_panel_rows)
        enrich_end = time.perf_counter()

        # ============ 4. Draw boxes and composite
        draw_start = time.perf_counter()
        np_img_with_boxes = videoHelper.draw_assigned_regions_on_frame(data["np_img"],
                                                                       list(shared_config.get('assigned_regions', [])))
        panel = videoHelper.build_detection_panel(enriched_panel_rows, data["np_img"].shape[0], )
        composite = videoHelper.side_by_side(np_img_with_boxes, panel)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, encoded_image = cv2.imencode('.jpg', composite, encode_param)
        jpeg_bytes = encoded_image.tobytes()
        draw_end = time.perf_counter()

        # ============ 5. Output FPS
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - last_time
        if elapsed >= 1:  # Update FPS every second
            fps = frame_count / elapsed
            frame_count = 0
            last_time = now
            shared_metrics.setdefault("output_fps", []).append(fps)
            shared_metrics["output_fps"][:] = shared_metrics["output_fps"][-N:]

        # ============ 6. Total processing time
        stage_end = time.perf_counter()

        # ============ 7. Send output
        try:
            mjpeg_frame_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            pass

        panel_time_ms = (panel_end - panel_start) * 1000
        shared_metrics.setdefault("output_panel_time_ms", []).append(panel_time_ms)
        shared_metrics["output_panel_time_ms"][:] = shared_metrics["output_panel_time_ms"][-N:]
        api_time_ms = (api_end - api_start) * 1000
        shared_metrics.setdefault("output_api_time_ms", []).append(api_time_ms)
        shared_metrics["output_api_time_ms"][:] = shared_metrics["output_api_time_ms"][-N:]
        draw_time_ms = (draw_end - draw_start) * 1000
        shared_metrics.setdefault("output_draw_time_ms", []).append(draw_time_ms)
        shared_metrics["output_draw_time_ms"][:] = shared_metrics["output_draw_time_ms"][-N:]
        enrich_time_ms = (enrich_end - enrich_start) * 1000
        shared_metrics.setdefault("output_panel_enrich_time_ms", []).append(enrich_time_ms)
        shared_metrics["output_panel_enrich_time_ms"][:] = shared_metrics["output_panel_enrich_time_ms"][-N:]
        total_output_time_ms = (stage_end - stage_start) * 1000
        shared_metrics.setdefault("output_total_processing_time_ms", []).append(total_output_time_ms)
        shared_metrics["output_total_processing_time_ms"][:] = shared_metrics["output_total_processing_time_ms"][-N:]

        input_queue.task_done()


def add_api_results_to_panel(panel_rows):
    """Add API results to panel rows for display"""
    enriched_rows = []

    for row in panel_rows:
        track_id = row["object_id"]
        enriched_row = row.copy()

        if track_id in api_results:
            api_data = api_results[track_id]
            enriched_row["api_result"] = {
                "code": api_data["code"],
                "confidence": api_data["confidence"]
            }

        enriched_rows.append(enriched_row)
    return enriched_rows


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


async def call_product_api_from_panel(panel_rows, frame_count, api_id_max_age=100):
    tasks = []
    current_frame_track_ids = set()
    for row in panel_rows:
        track_id = row["object_id"]
        current_frame_track_ids.add(track_id)
        if track_id in api_called_ids:
            continue
        ocr_results = row.get("ocr_results", [])
        ocr_text = " ".join([r["text"] for r in ocr_results if "text" in r])
        class_name = row["class_name"]
        if ocr_text.strip():
            tasks.append(fetch_product_id_async(class_name, ocr_text, track_id))
            api_called_ids.add(track_id)
        # Always update last seen for active IDs
        api_id_last_seen[track_id] = frame_count

    # Cleanup IDs not seen for > api_id_max_age frames
    to_remove = [tid for tid, last_seen in api_id_last_seen.items()
                 if frame_count - last_seen > api_id_max_age]
    for tid in to_remove:
        api_called_ids.discard(tid)
        api_id_last_seen.pop(tid, None)
        # Also cleanup API results
        api_results.pop(tid, None)

    if tasks:
        await asyncio.gather(*tasks)


async def fetch_product_id_async(class_name, ocr_text, track_id):
    import httpx
    url = "http://localhost:8001/api/product_lookup"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.post(url, json={"category": class_name, "ocr": ocr_text})
            if resp.status_code == 200:
                data = resp.json()
                code = data.get('code')
                confidence = data.get('confidence', 0.0)

                # Store the API result with timestamp
                api_results[track_id] = {
                    "code": code,
                    "confidence": confidence,
                    "timestamp": time.perf_counter()
                }

                print(f"[API] {class_name} + {ocr_text} → ProductID: {code}, conf: {confidence}")
            else:
                print(f"[API] Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[API] Exception: {e}")
