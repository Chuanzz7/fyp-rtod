import cv2


def draw(img, labels, boxes, scores, thrh=0.4):
    """
    Draw bounding boxes and labels on images using OpenCV (numpy arrays)

    Args:
        img: numpy arrays (H, W, C) in RGB format
        labels: List of label arrays for each image
        boxes: List of bounding box arrays for each image
        scores: List of confidence score arrays for each image
        thrh: Confidence threshold for filtering detections

    Returns:
        List of numpy arrays with drawn boxes and labels
    """
    img_copy = img.copy()

    scr = scores
    mask = scr > thrh
    lab = labels[mask]
    box = boxes[mask]
    scrs = scr[mask]

    for j, b in enumerate(box):
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, b)

        # Draw rectangle (BGR format for OpenCV)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)  # Red

        # Prepare text
        text = f"{lab[j].item()} {round(scrs[j].item(), 2)}"

        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw text background
        cv2.rectangle(img_copy, (x1, y1 - text_height - 10),
                      (x1 + text_width, y1), color=(255, 255, 255), thickness=-1)  # White background

        # Draw text
        cv2.putText(img_copy, text, (x1, y1 - 5),
                    font, font_scale, color=(255, 0, 0), thickness=thickness)  # Blue text

    return img_copy
