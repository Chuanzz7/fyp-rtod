from PIL import Image, ImageDraw, ImageFont

PANEL_W = 220  # width of the info sidebar
FONT = ImageFont.load_default()  # use a TTF of your choice

def build_detection_panel(results, height):
    """Return an RGB PIL image (PANEL_W × height) listing top detections."""
    panel = Image.new("RGB", (PANEL_W, height), (20, 20, 20))
    draw = ImageDraw.Draw(panel)

    y = 10
    for r in results:
        label = f"{r['class_name']:<12} {r['confidence'] * 100:5.1f}%"
        draw.text((10, y), label, fill=(255, 255, 255), font=FONT)
        y += 14
        if y > height - 14:  # stop if panel is full
            break
    return panel


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Horizontally concat two PIL images of identical height."""
    dst = Image.new("RGB", (left.width + right.width, left.height))
    dst.paste(left, (0, 0))
    dst.paste(right, (left.width, 0))
    return dst
