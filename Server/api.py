# Server/api.py
import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ---- local modules ----
from DFINE.detection import load_model, process_image  # <—— fixed

# ---- model paths ----
ROOT = Path(__file__).resolve().parent.parent  # project_root/
CONFIG_PATH = ROOT / "DFINE/configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml"
WEIGHTS_PATH = ROOT / "DFINE/model/dfine_x_obj365.pth"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH), DEVICE)  # load **once** at startup

app = FastAPI(title="D‑FINE Object Detection API")


# -----------------------------------------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        t0 = time.time()

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        img_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ---------- inference ----------
        results, annotated_image = process_image(model, DEVICE, image_pil)

        # ---------- save / encode ----------
        annotated_path = Path("output_annotated.jpg")
        annotated_image.save(annotated_path)

        response = []
        for idx, res in enumerate(results):
            crop_path = f"results/cropped_{res['class_name']}_{idx}.jpg"
            res["cropped_image"].save(crop_path)

            # make everything JSON‑serialisable
            confidence = float(res["confidence"])  # tensor → float
            box = [float(x) for x in res["box"]]  # tensor list → Python list

            response.append({
                "class_name": res["class_name"],
                "confidence": confidence,
                "box": box,
                "cropped_image_path": crop_path
            })

        # -------- delete the annotated file --------
        annotated_path.unlink(missing_ok=True)  # delete the file
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------
@app.get("/annotated_image")
async def get_annotated_image():
    file = Path("output_annotated.jpg")
    if not file.exists():
        raise HTTPException(status_code=404, detail="Annotated image not found")
    return FileResponse(file)


@app.get("/health")
def health():
    return {"status": "ok"}


# Class for the frame payload
class Frame(BaseModel):
    frame: str  # Base64 encoded image frame


# Endpoint for receiving video frames
@app.post("/upload_frame")
async def upload_frame(frame: Frame):
    try:
        # Decode the base64 image frame
        img_data = base64.b64decode(frame.frame)
        nparr = np.frombuffer(img_data, np.uint8)  # Convert bytes to NumPy array
        decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the frame into an image
        decoded_frame_rgb = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)

        # Display the frame (optional)
        cv2.imshow("Received Video", decoded_frame_rgb)

        # If you want to process or save the frame, you can do that here
        # For example, you could run inference or save the frame

        # You could use D-FINE or any other model to process the frame
        # Here is where you would call your model to process the `decoded_frame`

        # Check for a keypress to break out of the loop (useful for debugging)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return JSONResponse(content={"message": "Stream stopped"}, status_code=200)

        return JSONResponse(content={"message": "Frame received and processed"}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {e}")


# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
