import asyncio
import threading
import time
from typing import Dict, Any, List

import httpx

PRODUCT_ENDPOINT = "http://localhost:8001"
PI_URL = "http://192.168.0.93:9000"


class APIManager:
    """
    Manages asynchronous API calls in a dedicated background thread to avoid
    blocking the main processing loop. It is thread-safe.
    """

    def __init__(self, api_id_max_age: int = 100, base_url: str = PRODUCT_ENDPOINT, timeout: float = 1.0):
        """
        Initializes the APIManager.

        Args:
            api_id_max_age (int): Number of frames after which a non-visible track ID is cleaned up.
            base_url (str): The base URL for the API server.
            timeout (float): Timeout for API requests in seconds.
        """
        self.api_id_max_age = api_id_max_age

        # Thread-safe state, protected by the lock
        self._lock = threading.Lock()
        self.api_results: Dict[int, Dict[str, Any]] = {}
        self.api_id_last_seen: Dict[int, int] = {}

        # Setup for the background asyncio thread
        self._task_queue = asyncio.Queue()
        self._loop = asyncio.new_event_loop()
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

        self._runner_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._runner_thread.start()

    def _run_event_loop(self):
        """Target for the background thread. Runs the asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._task_consumer())

    async def _task_consumer(self):
        """Consumes and executes coroutine tasks from the queue."""
        while True:
            try:
                coro = await self._task_queue.get()
                if coro is None:  # Sentinel for shutdown
                    break
                # Creates a task that runs in the background of the event loop
                self._loop.create_task(coro)
            except Exception as e:
                print(f"[APIManager] Error in task consumer: {e}")

    def process_and_call_api(self, panel_rows: List[Dict[str, Any]], frame_count: int):
        """
        Identifies new objects that need an API call and submits them as a batch
        to the background queue without blocking. Also cleans up old track IDs.
        """
        batch_data = []
        current_frame_track_ids = {row["object_id"] for row in panel_rows}

        with self._lock:
            for row in panel_rows:
                track_id = row["object_id"]

                ocr_results = row.get("best_ocr_texts", [])
                ocr_text = " ".join(ocr_results)

                if ocr_text:
                    class_name = row["class_name"]
                    batch_data.append({
                        "track_id": track_id,
                        "category": class_name,
                        "ocr": ocr_text
                    })

            # Update last seen timestamp for all currently visible tracks
            for track_id in current_frame_track_ids:
                self.api_id_last_seen[track_id] = frame_count

            # Clean up old track IDs that haven't been seen for a while
            to_remove = [
                tid for tid, last_seen in self.api_id_last_seen.items()
                if frame_count - last_seen > self.api_id_max_age
            ]
            for tid in to_remove:
                # self.api_called_ids.discard(tid)
                self.api_id_last_seen.pop(tid, None)
                self.api_results.pop(tid, None)

            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._fetch_product_ids_batch_async(batch_data)))

    async def _fetch_product_ids_batch_async(self, batch_data: List[Dict[str, Any]]):
        """The actual async function that performs the batch API call."""
        try:
            # Send all items in a single API call
            resp = await self._client.post("/api/product_lookup_batch_monitor", json={"items": batch_data})

            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', [])

                # Process each result and store by track_id
                with self._lock:
                    for item_result in results:
                        track_id = item_result.get('track_id')
                        if track_id is not None and item_result.get('code') is not None:
                            result = {
                                "code": item_result.get('code', 'N/A'),
                                "confidence": item_result.get('confidence', 0.0),
                                "timestamp": time.perf_counter()
                            }
                            self.api_results[track_id] = result
            else:
                print(f"[API] Batch error {resp.status_code}: {resp.text}")
                track_ids = [item['track_id'] for item in batch_data]
                print(f"[API] Failed batch contained track_ids: {track_ids}")

        except Exception as e:
            track_ids = [item['track_id'] for item in batch_data]
            print(f"[API] Batch processing exception for track_ids {track_ids}: {e}")

    def get_api_results(self) -> Dict[int, Dict[str, Any]]:
        """Thread-safely returns a copy of the current API results."""
        with self._lock:
            return self.api_results.copy()

    def shutdown(self):
        """Gracefully shuts down the background thread and closes the client."""
        print("[APIManager] Shutting down...")
        # Send shutdown sentinel to the consumer
        self._loop.call_soon_threadsafe(self._task_queue.put_nowait, None)

        # Create a future to await client closing
        future = asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop)
        try:
            future.result(timeout=2)  # Wait for client to close
        except Exception as e:
            print(f"[APIManager] Error closing http client: {e}")

        # Stop the loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._runner_thread.join(timeout=2)
        print("[APIManager] Shutdown complete.")
