from typing import Tuple, List, Dict, Any

import cv2
import numpy as np

PANEL_W = 350  # Set as appropriate
PADDING_X = 18
PADDING_Y = 16
LINE_SPACING = 8
DETECTION_GAP = 16

BG_COLOR = (15, 15, 15)
TEXT_COLOR = (255, 255, 255)
OCR_COLOR = (140, 210, 255)
HIGH_CONF_COLOR = (110, 255, 110)
MID_CONF_COLOR = (255, 210, 80)
LOW_CONF_COLOR = (255, 100, 100)
API_COLOR = (255, 150, 255)  # Magenta for API results


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    if confidence > 0.8: return (110, 255, 110)  # High-conf green
    if confidence > 0.5: return (80, 210, 255)  # Mid-conf yellow
    return (100, 100, 255)  # Low-conf red


def get_status_color(status: str) -> Tuple[int, int, int]:
    if status == "Correct Region": return (110, 255, 110)  # Green
    if status.startswith("Wrong Region"): return (100, 100, 255)  # Red
    return (80, 210, 255)  # Yellow for "Unassigned"


def build_detection_panel(enriched_rows: List[Dict[str, Any]], height: int) -> np.ndarray:
    """
    Builds an optimized RGB numpy array with detection results using OpenCV.
    Assumes `enriched_rows` contains all data needed for display, including API results.
    """
    panel = np.full((height, PANEL_W, 3), BG_COLOR, dtype=np.uint8)
    y = PADDING_Y
    panel_bottom_margin = height - PADDING_Y

    # Font settings
    font_pri, font_sec = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX
    scale_pri, scale_sec = 0.6, 0.5
    thick_pri, thick_sec = 1, 1

    def draw_line(text, y_pos, font, scale, thick, color, indent=0):
        cv2.putText(panel, text, (PADDING_X + indent, y_pos), font, scale, color, thick, cv2.LINE_AA)

    for row in enriched_rows:
        if y + 60 > panel_bottom_margin:  # Estimate space for a full entry, break if not enough
            break

        current_y = y
        # Line 1: Class, ID, Confidence
        conf = row['confidence']
        label = f"ID:{row.get('object_id', 'N/A'):<3} {row['class_name']:<12} {conf * 100:5.1f}%"
        current_y += 20
        draw_line(label, current_y, font_pri, scale_pri, thick_pri, get_confidence_color(conf))

        # Line 2: Region Status
        status = row.get("region_status", "Unassigned")
        current_y += 20
        draw_line(status, current_y, font_pri, scale_pri, thick_pri, get_status_color(status))

        # Line 3: API Result (if present)
        if api_result := row.get("api_result"):
            code = str(api_result.get("code", "Unknown"))[:24]
            api_conf = api_result.get("confidence", 0.0)
            api_label = f"  API: {code} ({api_conf * 100:4.1f}%)"
            current_y += 18
            draw_line(api_label, current_y, font_sec, scale_sec, thick_sec, API_COLOR)

        # Lines 4+: OCR Results (if present)
        if ocr_results := row.get("best_ocr_texts"):
            for ocr_text in ocr_results:
                if not ocr_text: continue
                ocr_text_to_draw = ocr_text[:28]
                current_y += 18
                if current_y > panel_bottom_margin: break
                # Draw the text string directly
                draw_line(f"  -> {ocr_text_to_draw}", current_y, font_sec, scale_sec, thick_sec, OCR_COLOR, indent=12)

        # if ocr_results := row.get("ocr_results"):
        #     for ocr in ocr_results:
        #         ocr_text = ocr.get('text', '')[:28]
        #         if not ocr_text: continue
        #         current_y += 18
        #         if current_y > panel_bottom_margin: break
        #         draw_line(f"  -> {ocr_text}", current_y, font_sec, scale_sec, thick_sec, OCR_COLOR, indent=12)

        y = current_y + DETECTION_GAP

    return panel


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Horizontally concatenate two numpy arrays of identical height."""
    return np.hstack([left, right])


def draw_assigned_regions_on_frame(img_array, assigned_regions):
    for reg in assigned_regions:
        x1, y1, x2, y2 = map(int, reg["bbox"])
        label = reg["label"]

        # Choose color: left=blue, right=orange
        # OpenCV uses BGR format, so we need to convert RGB to BGR
        color_bgr = (255, 180, 80)  # Blue in BGR (was RGB 80, 180, 255)

        # Draw rectangle
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color=color_bgr, thickness=4)

        # Prepare text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Get text size for better positioning
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw text background for better readability
        text_x, text_y = x1 + 5, y1 + 5 + text_height
        cv2.rectangle(img_array,
                      (text_x - 2, text_y - text_height - 2),
                      (text_x + text_width + 2, text_y + baseline + 2),
                      color=(0, 0, 0), thickness=-1)  # Black background

        # Draw text
        cv2.putText(img_array, label, (text_x, text_y),
                    font, font_scale, color=color_bgr, thickness=thickness)

    return img_array
