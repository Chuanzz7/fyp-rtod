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


def build_detection_panel(results, height, region_statuses=None):
    """
    Return an RGB PIL image (PANEL_W × height) with:
    • Class name and detector confidence (large font)
    • Optional OCR text and its confidence (smaller font, colored)
    """

    panel = Image.new("RGB", (PANEL_W, height), BG_COLOR)
    draw = ImageDraw.Draw(panel)

    y = PADDING_Y
    for r in results:
        # Confidence color
        conf = r['confidence']
        if conf > 0.8:
            conf_color = HIGH_CONF_COLOR
        elif conf > 0.5:
            conf_color = MID_CONF_COLOR
        else:
            conf_color = LOW_CONF_COLOR

        # ── line-1 : Detector output with Track ID ──
        track_id = r.get('object_id', 'N/A')
        label = f"ID:{track_id:<3} {r['class_name']:<12} {conf * 100:5.1f}%"
        draw.text((PADDING_X, y), label, fill=conf_color, font=PRIMARY_FONT)
        y += PRIMARY_FONT.size + LINE_SPACING

        # ── line-2: Status results (if available) ──
        status = r.get("region_status", "Unassigned")
        if status == "Correct Region":
            status_color = (110, 255, 110)  # green
        elif status.startswith("Wrong Region"):
            status_color = (255, 100, 100)  # red
        else:
            status_color = (255, 210, 80)  # yellow

        label = f"{status}"
        draw.text((PADDING_X, y), label, fill=status_color, font=PRIMARY_FONT)
        y += PRIMARY_FONT.size + LINE_SPACING

        # ── line-3: OCR results (if available) ──
        if r.get("ocr_results"):
            for o in r["ocr_results"]:
                # Safe key access with fallbacks
                ocr_text = o.get('ocr_text', o.get('text', ''))
                ocr_conf = o.get('ocr_conf', o.get('ocr_confidence', o.get('confidence', 0)))

                if ocr_text:  # Only display if we have text
                    # Truncate with ellipsis
                    max_chars = 28
                    ocr_text = (ocr_text[:max_chars] + '…') if len(ocr_text) > max_chars else ocr_text
                    ocr_label = f"  -> {ocr_text} ({ocr_conf * 100:4.1f}%)"
                    draw.text((PADDING_X + 12, y), ocr_label, fill=OCR_COLOR, font=SECONDARY_FONT)
                    y += SECONDARY_FONT.size + LINE_SPACING // 2
                    if y > height - PADDING_Y - 20:
                        break

        # Gap before next detection
        y += DETECTION_GAP
        if y > height - PADDING_Y - 20:
            break

    return panel


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Horizontally concat two PIL images of identical height."""
    dst = Image.new("RGB", (left.width + right.width, left.height))
    dst.paste(left, (0, 0))
    dst.paste(right, (left.width, 0))
    return dst
