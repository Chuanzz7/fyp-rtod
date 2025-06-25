import asyncio
import time
from typing import List, Optional

import requests
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse, JSONResponse

from Detection.processor.processorSingleImage import SingleImageProcessor

# Allow your frontend origin, or use ["*"] for any (dev only!)
origins = [
    "*",
    # add more origins if needed, e.g. "http://127.0.0.1:5173"
]

app = FastAPI()
app.state.upload_counter = 0
app.state.last_logged = time.perf_counter()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)
PI_URL = "http://192.168.0.93:9000"


# Pydantic models for bounding box API
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class RegionUpdate(BaseModel):
    label: str
    bbox: BoundingBox
    type: str = "class"


class RegionsUpdate(BaseModel):
    regions: List[RegionUpdate]


# Updated configuration model to include assigned_regions
class ConfigUpdate(BaseModel):
    iou_threshold: Optional[float] = None
    api_id_max_age: Optional[int] = None
    jpeg_quality: Optional[int] = None
    fps_update_interval: Optional[float] = None
    font_size: Optional[int] = None
    assigned_regions: Optional[List[RegionUpdate]] = None


def inject_queues(frame_queue, mjpeg_queue, shared_config, shared_metrics):
    app.state.frame_input_queue = frame_queue
    app.state.mjpeg_frame_queue = mjpeg_queue
    app.state.shared_config = shared_config
    app.state.single_image_processor = SingleImageProcessor()
    app.state.shared_metrics = shared_metrics


async def qps_logger(interval=1):
    while True:
        await asyncio.sleep(interval)
        count = app.state.upload_counter
        app.state.upload_counter = 0
        qps = count / interval

        # Instead of just printing, record in shared_metrics
        shared_metrics = app.state.shared_metrics
        shared_metrics["upload_frame_qps"].append(qps)

        N = 120
        shared_metrics["upload_frame_qps"][:] = shared_metrics["upload_frame_qps"][-N:]


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


@app.get("/api/regions")
async def get_regions():
    """Get current assigned regions from shared config"""
    try:
        regions = list(app.state.shared_config['assigned_regions'])
        return {"regions": regions}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/regions")
async def update_regions(regions_update: RegionsUpdate):
    """Update assigned regions in shared config"""
    try:
        # Convert Pydantic models to the expected format
        new_regions = []
        for region in regions_update.regions:
            new_regions.append({
                "label": region.label,
                "bbox": [region.bbox.x1, region.bbox.y1, region.bbox.x2, region.bbox.y2],
                "type": region.type
            })

        # Update the shared config assigned_regions
        app.state.shared_config['assigned_regions'][:] = new_regions

        return {"status": "success", "message": f"Updated {len(new_regions)} regions"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/regions/single")
async def update_single_region(region_index: int, region_update: RegionUpdate):
    """Update a single region by index in shared config"""
    try:
        assigned_regions = app.state.shared_config['assigned_regions']

        if region_index < 0 or region_index >= len(assigned_regions):
            return {"status": "error", "message": "Invalid region index"}

        # Update the specific region
        assigned_regions[region_index] = {
            "label": region_update.label,
            "bbox": [region_update.bbox.x1, region_update.bbox.y1, region_update.bbox.x2, region_update.bbox.y2],
            "type": region_update.type
        }

        return {"status": "success", "message": f"Updated region {region_index}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/regions/{region_index}")
async def delete_region(region_index: int):
    """Delete a region by index from shared config"""
    try:
        assigned_regions = app.state.shared_config['assigned_regions']

        if region_index < 0 or region_index >= len(assigned_regions):
            return {"status": "error", "message": "Invalid region index"}

        # Remove the region
        del assigned_regions[region_index]

        return {"status": "success", "message": f"Deleted region {region_index}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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


@app.get("/api/metrics")
async def get_metrics():
    stats = {}
    for key, values in app.state.shared_metrics.items():
        if values:
            stats[key] = {
                "last": values[-1],
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
    return {"status": "success", "metrics": stats}
