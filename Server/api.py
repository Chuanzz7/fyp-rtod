# Server/api.py
import os
from pathlib import Path

from DFINE.tools.inference.trt_inf import TRTInference

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import queue
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from processor.processorTensor import process_loop

app = FastAPI(title="D‑FINE Object Detection API")
# limit queue to 30 frames to prevent backlog
MAX_QUEUE_SIZE = 30  # match TARGET_FPS
frame_queue = queue.Queue(MAX_QUEUE_SIZE)
jpeg_lock = threading.Lock()
latest_jpeg = {"frame": b""}

# ── paths & constants ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSOR_MODEL = PROJECT_ROOT / "DFINE" / "model" / "model.engine"
DEVICE = "cuda:0"
ENGINE = TRTInference(TENSOR_MODEL)

# start the background consumer
threading.Thread(
    target=process_loop,
    args=(frame_queue, ENGINE, DEVICE),
    daemon=True
).start()


@app.get("/health")
def health():
    return {"status": "ok"}


# ╭──────────────────────────────────────────────────────╮
# │ 1)  Receive, run inference, store annotated jpeg     │
# ╰──────────────────────────────────────────────────────╯
@app.post("/upload_frame")
async def upload_frame(request: Request):
    recv_ts = time.time()
    sent_ts = request.headers.get("X-Sent-Ts")
    if sent_ts:
        print(f" Network RTT: {(recv_ts - float(sent_ts)) * 1000:.1f} ms")

    data = await request.body()
    # drop oldest if queue full to keep only latest frames
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except queue.Empty:
            pass

    frame_id = time.time()
    frame_queue.put({
        "frame_id": frame_id,
        "data": data
    })
    return Response(status_code=200)


# @app.post("/upload_frame")
# async def upload_frame(request: Request):
#     global latest_jpeg
#     t_start = time.perf_counter()
#
#     try:
#
#         recv_ts = time.time()
#         sent_ts = request.headers.get("X-Sent-Ts")
#         if sent_ts:
#             print(f" Network RTT: {(recv_ts - float(sent_ts)) * 1000:.1f} ms")
#         t0 = time.perf_counter()
#
#         # 1) body read
#         jpg_bytes = await request.body()
#         t1 = time.perf_counter()
#         print(f"1) read body: {(t1 - t0) * 1000:.1f} ms")
#
#         # 2) JPEG → BGR decode
#         bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
#         if bgr is None:
#             raise HTTPException(400, "jpeg decode failed")
#         t2 = time.perf_counter()
#         print(f"2) decode: {(t2 - t1) * 1000:.1f} ms")
#
#         # ── 2. inference with D-FINE ───────────────────────────────────────
#         pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
#         labels, boxes, scores = torch_inf_pil(model, pil_img, DEVICE)
#         results = [
#             {
#                 "class_name": OBJECTS365_CLASSES[l.item() - 1][1],
#                 "confidence": s.item(),
#                 "box": b.tolist()
#             }
#             for l, b, s in zip(labels[0], boxes[0], scores[0]) if s > 0.4
#         ]
#         t3 = time.perf_counter()
#         print(f"3) inference: {(t3 - t2) * 1000:.1f} ms")
#
#         # now draw boxes for the video feed
#         annotated_pil = pil_img.copy()
#         draw([annotated_pil], labels, boxes, scores, thrh=0.8)
#
#         # --- build sidebar & compose final frame ----------------------------
#         panel_img = videoHelper.build_detection_panel(results, annotated_pil.height)
#         composite = videoHelper.side_by_side(annotated_pil, panel_img)
#         t4 = time.perf_counter()
#         print(f"4) draw+compose: {(t4 - t3) * 1000:.1f} ms")
#
#         buf = io.BytesIO()
#         composite.save(buf, format="JPEG", quality=85)
#         with jpeg_lock:
#             latest_jpeg = buf.getvalue()
#         t5 = time.perf_counter()
#         print(f"5) re-encode save: {(t5 - t4) * 1000:.1f} ms")
#
#         # ── 4. return just the detections summary ──────────────────────────
#         detections = [
#             {
#                 "class_name": r["class_name"],
#                 "confidence": float(r["confidence"]),
#                 "box": [float(x) for x in r["box"]],
#             } for r in results
#         ]
#         total = (time.perf_counter() - t_start) * 1000
#         print(f"✅ total handler: {total:.1f} ms")
#         return JSONResponse({"detections": detections}, status_code=200)
#
#     except Exception as e:
#         traceback.print_exc()  # prints full stack trace to stderr
#         raise HTTPException(status_code=500,
#                             detail=f"Error processing frame: {e}")


# ╭──────────────────────────────────────────────────────╮
# │ 2)  Live MJPEG stream anyone can watch at /video     │
# ╰──────────────────────────────────────────────────────╯
@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def mjpeg_generator():
    """Yields the newest annotated JPEG as a multipart/x-mixed-replace stream."""
    boundary = b"--frame\r\n"
    while True:
        with jpeg_lock:
            frame_jpeg = latest_jpeg
        if frame_jpeg is not None:
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame_jpeg.get('frame') + b"\r\n"
        time.sleep(0.025)  # ~20 fps push; no harm if slower upstream


@app.post("/ocr/")
def run_ocr(file: UploadFile):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    start_time = time.perf_counter()
    result = ocr.ocr(image, det=True, cls=True)
    end_time = time.perf_counter()

    print(f"OCR took {end_time - start_time:.3f} seconds")
    return result


# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
