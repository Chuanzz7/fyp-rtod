import multiprocessing
import os
import threading
import time
from multiprocessing import Process

import uvicorn

from Detection import api
from Detection.helper.dataClass import COCO_CLASSES
from Detection.processor import processorTensor, processorOutput

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def run_uvicorn():
    uvicorn.run("Detection.api:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    frame_input_queue = manager.Queue(maxsize=10)
    inference_output_queue = manager.Queue(maxsize=5)
    mjpeg_frame_queue = manager.Queue(maxsize=10)

    # Create shared configuration dictionary with assigned_regions
    shared_config = manager.dict({
        'assigned_regions': manager.list([
            {
                "label": COCO_CLASSES[39][1],  # user label, could be from OCR or model class
                "bbox": [0, 0, 320, 640],  # [x1, y1, x2, y2] in image coordinates
                "type": "class",  # "ocr" or "class" (how to compare)
            },
        ])
    })
    shared_metrics = manager.dict()
    shared_metrics["upload_frame_qps"] = manager.list()
    shared_metrics["decode_time_ms"] = manager.list()
    shared_metrics["dfine_inference_time_ms"] = manager.list()
    shared_metrics["sort_and_cache_time_ms"] = manager.list()
    shared_metrics["draw_time_ms"] = manager.list()
    shared_metrics["ocr_time_ms"] = manager.list()
    shared_metrics["total_processing_time_ms"] = manager.list()
    shared_metrics["output_panel_time_ms"] = manager.list()
    shared_metrics["output_api_time_ms"] = manager.list()
    shared_metrics["output_draw_time_ms"] = manager.list()
    shared_metrics["output_encode_time_ms"] = manager.list()
    shared_metrics["output_total_processing_time_ms"] = manager.list()
    shared_metrics["output_fps"] = manager.list()

    p1 = Process(target=processorTensor.processor_tensor_main,
                 args=(frame_input_queue, inference_output_queue, shared_metrics))
    p2 = Process(target=processorOutput.process_output_main,
                 args=(inference_output_queue, mjpeg_frame_queue, shared_config, shared_metrics))

    p1.start()
    p2.start()

    # Run API and inject queues and shared config
    api.inject_queues(frame_input_queue, mjpeg_frame_queue, shared_config, shared_metrics)

    # Start Uvicorn in a thread so the main process isn't blocked
    uvicorn_thread = threading.Thread(target=run_uvicorn, daemon=True)
    uvicorn_thread.start()

    print("Detection boot complete, inference ready and standing by for frames.")

    # Keep main thread alive (since all logic is async/threaded/multiproc)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

    p1.terminate()
    p2.terminate()
    p1.join()
    p2.join()
