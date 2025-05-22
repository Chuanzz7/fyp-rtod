import asyncio

import requests
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

app = FastAPI()


def inject_queues(frame_queue, mjpeg_queue):
    app.state.frame_input_queue = frame_queue
    app.state.mjpeg_frame_queue = mjpeg_queue


@app.post("/upload_frame")
async def upload_frame(request: Request):
    frame_bytes = await request.body()
    try:
        app.state.frame_input_queue.put_nowait(frame_bytes)
    except Exception as e:
        return {"status": f"queue error: {e}"}
    return {"status": "frame received"}


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


@app.get("/activeMonitoring")
def start():
    # Wait for the API to be up before sending /start_stream
    # Now tell the Pi to start sending frames
    PI_URL = "http://192.168.0.92:9000"
    try:
        resp = requests.post(f"{PI_URL}/start_stream", timeout=2)
        print("Pi response:", resp.text)
    except Exception as e:
        print("Failed to contact Pi:", e)
