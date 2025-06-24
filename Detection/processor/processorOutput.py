import asyncio
import io
import queue
import time
from multiprocessing import Queue

from PIL import ImageDraw, ImageFont

from Detection.helper import videoHelper

api_called_ids = set()
api_id_last_seen = dict()

# Remove the global ASSIGNED_REGIONS since it's now in shared config


def process_output_main(input_queue: Queue, mjpeg_frame_queue: Queue, shared_config):
    last_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        try:
            data = input_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Get current configuration values including assigned_regions
        iou_threshold = shared_config.get('iou_threshold', 0.1)
        api_id_max_age = shared_config.get('api_id_max_age', 100)
        jpeg_quality = shared_config.get('jpeg_quality', 85)
        fps_update_interval = shared_config.get('fps_update_interval', 1.0)
        font_size = shared_config.get('font_size', 28)
        assigned_regions = list(shared_config.get('assigned_regions', []))

        # Use dynamic font size
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Compute placement status for all detections with dynamic IoU threshold and regions
        checked_panel_rows = check_wrong_placement(
            data["panel_rows"], assigned_regions, iou_threshold=iou_threshold
        )

        # --- API CALL: Call only once per object/track_id with OCR text ---
        asyncio.run(call_product_api_from_panel(checked_panel_rows, frame_count, api_id_max_age))
        # ---------------------------------------------------------------

        pil_img_with_boxes = draw_assigned_regions_on_frame(data["pil_img"], assigned_regions)

        # === FPS Calculation with dynamic interval ===
        frame_count += 1
        now = time.time()
        elapsed = now - last_time
        if elapsed >= fps_update_interval:
            fps = frame_count / elapsed
            frame_count = 0
            last_time = now

        # === Overlay FPS on the image ===
        draw = ImageDraw.Draw(pil_img_with_boxes)
        text = f"FPS: {fps:.2f}"
        draw.rectangle((10, 10, 140, 45), fill=(0, 0, 0, 127))  # semi-transparent background
        draw.text((15, 15), text, fill=(255, 255, 0), font=font)  # yellow text

        # Draw detection panel with placement info
        panel = videoHelper.build_detection_panel(
            checked_panel_rows, pil_img_with_boxes.height,
        )

        composite = videoHelper.side_by_side(pil_img_with_boxes, panel)
        buf = io.BytesIO()
        composite.save(buf, format="JPEG", quality=jpeg_quality)
        jpeg_bytes = buf.getvalue()

        try:
            mjpeg_frame_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            pass

        input_queue.task_done()


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


def draw_assigned_regions_on_frame(pil_img, assigned_regions):
    """Draw assigned region bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(pil_img)
    for reg in assigned_regions:
        x1, y1, x2, y2 = reg["bbox"]
        label = reg["label"]
        # Choose color: left=blue, right=orange (customizable)
        color = (80, 180, 255) if x1 == 0 else (255, 180, 80)
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        # Draw label (top-left of region)
        draw.text((x1 + 5, y1 + 5), label, fill=color)
    return pil_img


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
            tasks.append(fetch_product_id_async(class_name, ocr_text))
            api_called_ids.add(track_id)
        # Always update last seen for active IDs
        api_id_last_seen[track_id] = frame_count

    # Cleanup IDs not seen for > api_id_max_age frames
    to_remove = [tid for tid, last_seen in api_id_last_seen.items()
                 if frame_count - last_seen > api_id_max_age]
    for tid in to_remove:
        api_called_ids.discard(tid)
        api_id_last_seen.pop(tid, None)

    if tasks:
        await asyncio.gather(*tasks)

async def fetch_product_id_async(class_name, ocr_text):
    import httpx
    url = "http://localhost:8001/api/product_lookup"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.post(url, json={"category": class_name, "ocr": ocr_text})
            if resp.status_code == 200:
                data = resp.json()
                print(
                    f"[API] {class_name} + {ocr_text} → ProductID: {data.get('product_id')}, conf: {data.get('confidence')}")
            else:
                print(f"[API] Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[API] Exception: {e}")