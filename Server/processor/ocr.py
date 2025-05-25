import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from paddleocr import PaddleOCR, draw_ocr
import cv2

# Setup OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Image path
img_path = r'IMG_5023.jpg'
print("Image exists?", os.path.isfile(img_path))

# Measure time
start_time = time.perf_counter()
result = ocr.ocr(img_path, det=True, cls=True)
end_time = time.perf_counter()

print(f"OCR took {end_time - start_time:.3f} seconds")

# Print OCR results
for res in result:
    for line in res:
        print(line)

# Optional: save result image
img = cv2.imread(img_path)
boxes = [line[0] for res in result for line in res]
txts = [line[1][0] for res in result for line in res]
scores = [line[1][1] for res in result for line in res]

img_out = draw_ocr(img, boxes, txts, scores)
cv2.imwrite("ocr_result.jpg", img_out)
