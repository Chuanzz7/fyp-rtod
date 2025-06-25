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


def get_confidence_color(confidence):
    """Returns the BGR color based on a confidence score."""
    if confidence > 0.8:
        return HIGH_CONF_COLOR
    if confidence > 0.5:
        return MID_CONF_COLOR
    return LOW_CONF_COLOR


def get_status_color(status):
    """Returns the BGR color based on a region status string."""
    if status == "Correct Region":
        return (110, 255, 110)  # green (BGR)
    if status.startswith("Wrong Region"):
        return (100, 100, 255)  # red (BGR)
    return (80, 210, 255)  # yellow (BGR) - Unassigned or other


def build_detection_panel(results, height):
    """
    Return an optimized RGB numpy array with detection results using OpenCV.
    """
    # Create blank panel (BGR format for OpenCV)
    panel = np.full((height, PANEL_W, 3), BG_COLOR, dtype=np.uint8)

    # Font settings
    primary_font = cv2.FONT_HERSHEY_SIMPLEX
    secondary_font = cv2.FONT_HERSHEY_SIMPLEX
    primary_font_scale = 0.6
    secondary_font_scale = 0.5
    primary_thickness = 1
    secondary_thickness = 1

    # Calculate line heights
    primary_line_height = cv2.getTextSize("Tg", primary_font, primary_font_scale, primary_thickness)[0][
                              1] + LINE_SPACING
    secondary_line_height = cv2.getTextSize("Tg", secondary_font, secondary_font_scale, secondary_thickness)[0][
                                1] + LINE_SPACING // 2

    y = PADDING_Y
    panel_bottom_margin = height - PADDING_Y

    # --- Drawing Helper ---
    def draw_line(text, font, font_scale, thickness, color, indent=0, line_height=primary_line_height):
        nonlocal y
        if y + line_height > panel_bottom_margin:
            return False  # Not enough space to draw

        # Draw text on panel
        cv2.putText(panel, text, (PADDING_X + indent, y + line_height - 5),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height
        return True  # Successfully drawn

    # --- Main Loop ---
    for r in results:
        # ── Line 1: Detector output ──
        conf = r['confidence']
        track_id = r.get('object_id', 'N/A')
        label = f"ID:{track_id:<3} {r['class_name']:<12} {conf * 100:5.1f}%"
        if not draw_line(label, primary_font, primary_font_scale, primary_thickness,
                         get_confidence_color(conf)):
            break

        # ── Line 2: Status results ──
        status = r.get("region_status", "Unassigned")
        if not draw_line(f"{status}", primary_font, primary_font_scale, primary_thickness,
                         get_status_color(status)):
            break

        # ── Line 3: API results ──
        if api_data := r.get("api_result"):  # Walrus operator for cleaner code
            code = api_data.get("code", "Unknown")
            api_conf = api_data.get("confidence", 0.0)
            display_id = str(code)
            if len(display_id) > 24:
                display_id = display_id[:24] + '…'

            api_label = f"  API: {display_id} ({api_conf * 100:4.1f}%)"
            if not draw_line(api_label, secondary_font, secondary_font_scale, secondary_thickness,
                             API_COLOR, indent=0, line_height=secondary_line_height):
                break

        # ── Line 4: OCR results ──
        if ocr_results := r.get("ocr_results"):
            for o in ocr_results:
                ocr_text = o.get('ocr_text', o.get('text', ''))
                if not ocr_text:
                    continue

                ocr_conf = o.get('ocr_conf', o.get('ocr_confidence', 0))
                if len(ocr_text) > 28:
                    ocr_text = ocr_text[:28] + '…'

                ocr_label = f"  -> {ocr_text} ({ocr_conf * 100:4.1f}%)"
                if not draw_line(ocr_label, secondary_font, secondary_font_scale, secondary_thickness,
                                 OCR_COLOR, indent=12, line_height=secondary_line_height):
                    # Set a flag to break the outer loop as well
                    y = height
                    break
            if y >= height: break  # Break from the main results loop

        y += DETECTION_GAP
        if y >= panel_bottom_margin:
            break

    # Convert from BGR to RGB for final output
    panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    return panel_rgb


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Horizontally concatenate two numpy arrays of identical height.

    Args:
        left: numpy array (H, W, C) - left image
        right: numpy array (H, W, C) - right image

    Returns:
        numpy array with horizontally concatenated images
    """
    # Use numpy's horizontal stack for efficient concatenation
    return np.hstack([left, right])


def draw_assigned_regions_on_frame(img_array, assigned_regions):
    """
    Draw assigned region bounding boxes and labels on the image using OpenCV.

    Args:
        img_array: numpy array (H, W, C) in RGB format
        assigned_regions: list of dictionaries with 'bbox' and 'label' keys

    Returns:
        numpy array with drawn regions
    """
    img_copy = img_array.copy()

    for reg in assigned_regions:
        x1, y1, x2, y2 = map(int, reg["bbox"])
        label = reg["label"]

        # Choose color: left=blue, right=orange
        # OpenCV uses BGR format, so we need to convert RGB to BGR
        if x1 == 0:
            color_bgr = (255, 180, 80)  # Blue in BGR (was RGB 80, 180, 255)
        else:
            color_bgr = (80, 180, 255)  # Orange in BGR (was RGB 255, 180, 80)

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color=color_bgr, thickness=4)

        # Prepare text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Get text size for better positioning
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw text background for better readability
        text_x, text_y = x1 + 5, y1 + 5 + text_height
        cv2.rectangle(img_copy,
                      (text_x - 2, text_y - text_height - 2),
                      (text_x + text_width + 2, text_y + baseline + 2),
                      color=(0, 0, 0), thickness=-1)  # Black background

        # Draw text
        cv2.putText(img_copy, label, (text_x, text_y),
                    font, font_scale, color=color_bgr, thickness=thickness)

    return img_copy
