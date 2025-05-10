# Server/api.py
import base64
import io
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---- local modules ----
from DFINE.tools.inference.torch_inf import YAMLConfig, OBJECTS365_CLASSES, draw
from Server.helper import videoHelper


# ────────────────────────────────────────────────────────────────────────────

# ─── build & warm-load the D-FINE model exactly the way torch_inf does ─────
def build_dfine_model(cfg_path: str, ckpt_path: str, device: str = "cpu"):
    """Replicates the Model() logic from torch_inf, but returns a ready model."""
    cfg = YAMLConfig(cfg_path, resume=ckpt_path)

    # stop HGNetv2 layers from re-downloading ImageNet weights
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, imgs, orig_sizes):
            return self.post(self.model(imgs), orig_sizes)

    return DeployModel().to(device).eval()


# ─── single-image inference that works on a PIL.Image (not a file-path) ────
_resize_to = (640, 640)
_tf = T.Compose([T.Resize(_resize_to), T.ToTensor()])


def torch_inf_pil(model, pil_img: Image.Image, device: str):
    """Return labels, boxes, scores → identical to detection.py output."""
    w, h = pil_img.size
    img_tensor = _tf(pil_img).unsqueeze(0).to(device)
    labels, boxes, scores = model(img_tensor, torch.tensor([[w, h]], device=device))
    return labels, boxes, scores


# ---- model paths ----
MODEL_CFG = "../DFINE/configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml"
MODEL_WEI = "../DFINE/model/dfine_x_obj365.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model = build_dfine_model(str(MODEL_CFG), str(MODEL_WEI), DEVICE)

app = FastAPI(title="D‑FINE Object Detection API")

# global holder for the latest annotated JPEG
latest_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()


# Class for the frame payload
class Frame(BaseModel):
    frame: str  # Base64 encoded image frame


@app.get("/health")
def health():
    return {"status": "ok"}


# ╭──────────────────────────────────────────────────────╮
# │ 1)  Receive, run inference, store annotated jpeg     │
# ╰──────────────────────────────────────────────────────╯
@app.post("/upload_frame")
async def upload_frame(frame: Frame):
    global latest_jpeg
    try:
        # ── 1. decode incoming base64 JPEG → OpenCV BGR ───────────────────
        img_bytes = base64.b64decode(frame.frame)
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ── 2. inference with D-FINE ───────────────────────────────────────
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        labels, boxes, scores = torch_inf_pil(model, pil_img, DEVICE)
        results = [
            {
                "class_name": OBJECTS365_CLASSES[l.item() - 1][1],
                "confidence": s.item(),
                "box": b.tolist()
            }
            for l, b, s in zip(labels[0], boxes[0], scores[0]) if s > 0.4
        ]
        # now draw boxes for the video feed
        annotated_pil = pil_img.copy()
        draw([annotated_pil], labels, boxes, scores, thrh=0.8)

        # --- build sidebar & compose final frame ----------------------------
        panel_img = videoHelper.build_detection_panel(results, annotated_pil.height)
        composite = videoHelper.side_by_side(annotated_pil, panel_img)

        buf = io.BytesIO()
        composite.save(buf, format="JPEG", quality=85)
        with jpeg_lock:
            latest_jpeg = buf.getvalue()

        # ── 4. return just the detections summary ──────────────────────────
        detections = [
            {
                "class_name": r["class_name"],
                "confidence": float(r["confidence"]),
                "box": [float(x) for x in r["box"]],
            } for r in results
        ]
        return JSONResponse({"detections": detections}, status_code=200)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500,
                            detail=f"Error processing frame: {e}")


# ╭──────────────────────────────────────────────────────╮
# │ 2)  Live MJPEG stream anyone can watch at /video     │
# ╰──────────────────────────────────────────────────────╯
def mjpeg_generator():
    """Yields the newest annotated JPEG as a multipart/x-mixed-replace stream."""
    boundary = b"--frame\r\n"
    while True:
        with jpeg_lock:
            frame_jpeg = latest_jpeg
        if frame_jpeg is not None:
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame_jpeg + b"\r\n"
        time.sleep(0.05)  # ~20 fps push; no harm if slower upstream


@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
