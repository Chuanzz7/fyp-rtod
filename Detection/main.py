import multiprocessing
import os
import threading
import time
from multiprocessing import Process

import uvicorn

from Detection import api
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

    p1 = Process(target=processorTensor.processor_tensor_main, args=(frame_input_queue, inference_output_queue))
    p2 = Process(target=processorOutput.process_output_main, args=(inference_output_queue, mjpeg_frame_queue))

    p1.start()
    p2.start()

    # Run API and inject queues
    # You can set api.py's queues here by importing or via global vars before uvicorn.run
    api.inject_queues(frame_input_queue, mjpeg_frame_queue)

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
