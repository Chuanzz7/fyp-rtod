import asyncio
import time

import requests
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse

from Detection.processor.processorSingleImage import SingleImageProcessor

# Allow your frontend origin, or use ["*"] for any (dev only!)
origins = [
    "*",
    # add more origins if needed, e.g. "http://127.0.0.1:5173"
]

app = FastAPI()
app.state.upload_counter = 0
app.state.last_logged = time.time()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)
PI_URL = "http://192.168.0.93:9000"

def inject_queues(frame_queue, mjpeg_queue):
    app.state.frame_input_queue = frame_queue
    app.state.mjpeg_frame_queue = mjpeg_queue
    app.state.single_image_processor = SingleImageProcessor()


async def qps_logger(interval=60):
    while True:
        await asyncio.sleep(interval)
        count = app.state.upload_counter
        app.state.upload_counter = 0
        print(f"[UPLOAD_FRAME QPS] {count} requests in last {interval} seconds | Avg: {count / interval:.2f} rps")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(qps_logger())


@app.post("/upload_frame")
async def upload_frame(request: Request):
    app.state.upload_counter += 1  # <--- increment counter
    # Get raw bytes
    frame_bytes = await request.body()
    try:
        app.state.frame_input_queue.put_nowait(frame_bytes)
    except Exception as e:
        return {"status": f"queue error: {e}"}
    return {"status": "frame received"}


@app.post("/api/detect_item")
async def upload_frame(image: UploadFile = File(...)):
    frame_bytes = await image.read()
    result = app.state.single_image_processor.process_image(
        image_input=frame_bytes,
        include_ocr=True,
        detection_threshold=0.8,
        ocr_threshold=0.5
    )
    return JSONResponse(content=result)


async def mjpeg_generator():
    boundary = "frameboundary"
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Run blocking get() in a thread pool
            frame = await loop.run_in_executor(None, app.state.mjpeg_frame_queue.get, True, 1)
            yield (
                    b"--" + boundary.encode() + b"\r\n"
                                                b"Content-Type: image/jpeg\r\n"
                                                b"Content-Length: " + f"{len(frame)}".encode() + b"\r\n\r\n" +
                    frame + b"\r\n"
            )
        except Exception:
            continue


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frameboundary"
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/start")
def start():
    # Wait for the API to be up before sending /start_stream
    # Now tell the Pi to start sending frames
    try:
        resp = requests.post(f"{PI_URL}/start_stream", timeout=2)
        print("Pi response:", resp.text)
    except Exception as e:
        print("Failed to contact Pi:", e)


@app.post("/stop")
def start():
    # Wait for the API to be up before sending /start_stream
    # Now tell the Pi to start sending frames
    try:
        resp = requests.post(f"{PI_URL}/stop_stream", timeout=2)
        print("Pi response:", resp.text)
    except Exception as e:
        print("Failed to contact Pi:", e)
