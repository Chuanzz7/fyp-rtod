from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request

from DFINE.tools.inference.trt_inf import TRTInference
from Server.profile_inference import profile_inference

# ── paths & constants ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSOR_MODEL = PROJECT_ROOT / "DFINE" / "model" / "model.engine"
DEVICE = "cuda:0"
ENGINE = TRTInference(TENSOR_MODEL)
app = FastAPI(title="D‑FINE Object Detection API")


def result_to_json(result):
    # This depends on D-FINE output format
    # For example:
    boxes = result["boxes"].cpu().numpy().tolist()
    scores = result["scores"].cpu().numpy().tolist()
    classes = result["labels"].cpu().numpy().tolist()
    return {"boxes": boxes, "scores": scores, "classes": classes}


@app.post("/upload_frame")
async def upload_frame(request: Request):
    data = await request.body()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("  !! JPEG decode failed – skipping frame")
        return {"error": "decode_failed"}

    # Run D-FINE inference
    result = profile_inference(ENGINE, img, device=DEVICE)

    # Optionally: Postprocess/convert result to JSON
    result_json = result_to_json(result)  # Implement this as needed
    return {"result": result_json}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
