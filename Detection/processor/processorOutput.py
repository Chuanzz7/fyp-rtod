import io
import queue
import time
from multiprocessing import Queue

from PIL import ImageDraw, ImageFont

from Detection.helper import videoHelper
from Detection.helper.dataClass import COCO_CLASSES

ASSIGNED_REGIONS = [
    {
        "label": COCO_CLASSES[39][1],  # user label, could be from OCR or model class
        "bbox": [0, 0, 320, 640],  # [x1, y1, x2, y2] in image coordinates
        "type": "class",  # "ocr" or "class" (how to compare)
    },
]


def process_output_main(input_queue: Queue, mjpeg_frame_queue: Queue):
    last_time = time.time()
    frame_count = 0
    fps = 0

    font = ImageFont.truetype("arial.ttf", 28)  # Choose size to taste

    while True:
        try:
            data = input_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Compute placement status for all detections
        checked_panel_rows = check_wrong_placement(
            data["panel_rows"], ASSIGNED_REGIONS
        )

        pil_img_with_boxes = draw_assigned_regions_on_frame(data["pil_img"], ASSIGNED_REGIONS)

        # === FPS Calculation ===
        frame_count += 1
        now = time.time()
        elapsed = now - last_time
        if elapsed >= 1.0:
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
        composite.save(buf, format="JPEG", quality=85)
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

    Updated to work with new flattened data structure where OCR text is directly in row
    """
    results = []

    for row in panel_rows:
        det_bbox = row["box"]
        det_label = row["class_name"]

        # Handle new flattened structure - OCR text is directly in the row
        ocr_texts = []
        if "ocr_text" in row and row["ocr_text"]:
            # Single OCR text field
            ocr_texts = [row["ocr_text"].lower()]
        elif "ocr_results" in row and row["ocr_results"]:
            # Backward compatibility - if ocr_results still exists as nested structure
            ocr_texts = [o["ocr_text"].lower() for o in row["ocr_results"]
                         if isinstance(o, dict) and "ocr_text" in o and o["ocr_text"]]
        elif "text" in row and row["text"]:
            # Alternative field name for OCR text
            ocr_texts = [row["text"].lower()]

        status = "Unassigned"
        matched_region_label = None

        # Check against each assigned region
        for reg in assigned_regions:
            reg_bbox = reg["bbox"]
            iou = compute_iou(det_bbox, reg_bbox)

            if iou > iou_threshold:
                matched_region_label = reg["label"]

                # Class-based region matching
                if reg["type"] == "class":
                    if det_label.lower() == reg["label"].lower():
                        status = "Correct Region"
                    else:
                        status = f"Wrong Region ({reg['label']})"

                # OCR-based region matching
                elif reg["type"] == "ocr":
                    # Check if any OCR text contains the region label
                    if ocr_texts and any(reg["label"].lower() in t for t in ocr_texts):
                        status = "Correct Region"
                    else:
                        status = f"Wrong Region ({reg['label']})"

                break  # Stop after first region overlap

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
