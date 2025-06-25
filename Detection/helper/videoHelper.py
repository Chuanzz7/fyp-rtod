from PIL import Image, ImageDraw, ImageFont

PANEL_W = 350  # Set as appropriate
PADDING_X = 18
PADDING_Y = 16
LINE_SPACING = 8
DETECTION_GAP = 16
PRIMARY_FONT = ImageFont.truetype("arial.ttf", 22)  # Adjust font path and size
SECONDARY_FONT = ImageFont.truetype("arial.ttf", 16)  # For OCR text

BG_COLOR = (15, 15, 15)
TEXT_COLOR = (255, 255, 255)
OCR_COLOR = (140, 210, 255)
HIGH_CONF_COLOR = (110, 255, 110)
MID_CONF_COLOR = (255, 210, 80)
LOW_CONF_COLOR = (255, 100, 100)
API_COLOR = (255, 150, 255)  # Magenta for API results


def get_confidence_color(confidence):
    """Returns the color based on a confidence score."""
    if confidence > 0.8:
        return HIGH_CONF_COLOR
    if confidence > 0.5:
        return MID_CONF_COLOR
    return LOW_CONF_COLOR


def get_status_color(status):
    """Returns the color based on a region status string."""
    if status == "Correct Region":
        return (110, 255, 110)  # green
    if status.startswith("Wrong Region"):
        return (255, 100, 100)  # red
    return (255, 210, 80)  # yellow (Unassigned or other)


def build_detection_panel(results, height):
    """
    Return an optimized RGB PIL image with detection results.
    """
    panel = Image.new("RGB", (PANEL_W, height), BG_COLOR)
    draw = ImageDraw.Draw(panel)

    # --- Pre-calculate font metrics ---
    # Use getbbox to get more accurate line heights than .size
    primary_line_height = PRIMARY_FONT.getbbox("Tg")[3] + LINE_SPACING
    secondary_line_height = SECONDARY_FONT.getbbox("Tg")[3] + LINE_SPACING // 2

    y = PADDING_Y
    panel_bottom_margin = height - PADDING_Y

    # --- Drawing Helper ---
    def draw_line(text, font, color, indent=0, line_height=primary_line_height):
        nonlocal y
        if y + line_height > panel_bottom_margin:
            return False  # Not enough space to draw

        draw.text((PADDING_X + indent, y), text, fill=color, font=font)
        y += line_height
        return True  # Successfully drawn

    # --- Main Loop ---
    for r in results:
        # ── Line 1: Detector output ──
        conf = r['confidence']
        track_id = r.get('object_id', 'N/A')
        label = f"ID:{track_id:<3} {r['class_name']:<12} {conf * 100:5.1f}%"
        if not draw_line(label, PRIMARY_FONT, get_confidence_color(conf)):
            break

        # ── Line 2: Status results ──
        status = r.get("region_status", "Unassigned")
        if not draw_line(f"{status}", PRIMARY_FONT, get_status_color(status)):
            break

        # ── Line 3: API results ──
        if api_data := r.get("api_result"):  # Walrus operator for cleaner code
            code = api_data.get("code", "Unknown")
            api_conf = api_data.get("confidence", 0.0)
            display_id = str(code)
            if len(display_id) > 24:
                display_id = display_id[:24] + '…'

            api_label = f"  API: {display_id} ({api_conf * 100:4.1f}%)"
            if not draw_line(api_label, SECONDARY_FONT, API_COLOR, indent=0, line_height=secondary_line_height):
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
                if not draw_line(ocr_label, SECONDARY_FONT, OCR_COLOR, indent=12, line_height=secondary_line_height):
                    # Set a flag to break the outer loop as well
                    y = height
                    break
            if y >= height: break  # Break from the main results loop

        y += DETECTION_GAP
        if y >= panel_bottom_margin:
            break

    return panel


# The side_by_side function is already optimal. No changes needed.
def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Horizontally concat two PIL images of identical height."""
    dst = Image.new("RGB", (left.width + right.width, left.height))
    dst.paste(left, (0, 0))
    dst.paste(right, (left.width, 0))
    return dst