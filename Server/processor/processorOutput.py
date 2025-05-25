import io
import queue
import time
from multiprocessing import Queue

from PIL import ImageDraw, ImageFont

from Server.helper import videoHelper
from Server.helper.dataClass import COCO_CLASSES

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


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


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
        ocr_texts = [o["ocr_text"].lower() for o in row.get("ocr_results", [])]
        status = "Unassigned"
        matched_region_label = None

        for reg in assigned_regions:
            reg_bbox = reg["bbox"]
            iou = compute_iou(det_bbox, reg_bbox)
            if iou > iou_threshold:
                matched_region_label = reg["label"]
                # Class mode
                if reg["type"] == "class":
                    if det_label.lower() == reg["label"].lower():
                        status = "Correct Region"
                    else:
                        status = f"Wrong Region ({reg['label']})"
                # OCR mode
                elif reg["type"] == "ocr":
                    if any(reg["label"].lower() in t for t in ocr_texts):
                        status = "Correct Region"
                    else:
                        status = f"Wrong Region ({reg['label']})"
                break  # Stop after first region overlap

        results.append({
            **row,  # keep all original keys for your panel code
            "region_status": status,
            "matched_region": matched_region_label,
        })
    return results


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
