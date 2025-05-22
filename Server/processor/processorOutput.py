import io
import queue
from multiprocessing import Queue

from Server.helper import videoHelper


def process_output_main(input_queue: Queue, mjpeg_frame_queue: Queue):
    while True:
        try:
            data = input_queue.get(timeout=1)
        except queue.Empty:
            continue

        panel = videoHelper.build_detection_panel(data["panel_rows"], data["pil_img"].height)
        composite = videoHelper.side_by_side(data["pil_img"], panel)
        buf = io.BytesIO()
        composite.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        try:
            mjpeg_frame_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            pass  # drop frame if queue full

        input_queue.task_done()
