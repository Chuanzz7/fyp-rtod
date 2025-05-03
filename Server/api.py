# Server/api.py
import io
import time
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse

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


# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
