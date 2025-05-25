"""
Enhanced profiling for D-FINE TensorRT inference to identify performance bottlenecks
"""

import time
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as T

# Keep your existing transforms
_tf = T.Compose([T.Resize((640, 640), interpolation=T.InterpolationMode.BILINEAR),
                 T.ToTensor()])  # still fastest on CPU


def profile_inference(engine, img_path, device="cuda:0", warmup_runs=5, profile_runs=10):
    """
    Detailed profiling of each step in the inference pipeline.

    Args:
        engine: The TensorRT inference engine
        img_path: Path to an image file for testing
        device: CUDA device
        warmup_runs: Number of warm-up iterations before timing
        profile_runs: Number of iterations to average timing over
    """
    # Load test image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return

    # Pre-allocate tensors to minimize allocations during timing
    img_tensor = torch.empty((1, 3, 640, 640), dtype=torch.float32, device=device)
    orig_size = torch.empty((1, 2), dtype=torch.int32, device=device)

    # Profile stages
    timings = {
        "decode_convert": [],
        "resize_transform": [],
        "to_device": [],
        "inference": [],
        "synchronize": [],
        "total": []
    }

    # Warm-up runs (not timed)
    print(f"Running {warmup_runs} warm-up iterations...")
    for _ in range(warmup_runs):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        w, h = pil_img.size
        tensor = _tf(pil_img).unsqueeze(0).to(device)
        orig_size[0, 0] = w
        orig_size[0, 1] = h

        blob = {
            "images": tensor,
            "orig_target_sizes": orig_size,
        }

        _ = engine(blob)
        torch.cuda.synchronize()

    # Actual profiling runs
    print(f"Running {profile_runs} profiling iterations...")
    for i in range(profile_runs):
        # Full pipeline timing
        t_start_total = time.perf_counter()

        # 1. Decode and convert
        t1 = time.perf_counter()
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        w, h = pil_img.size
        t2 = time.perf_counter()
        timings["decode_convert"].append((t2 - t1) * 1000)

        # 2. Resize and transform
        t1 = time.perf_counter()
        tensor = _tf(pil_img).unsqueeze(0)
        t2 = time.perf_counter()
        timings["resize_transform"].append((t2 - t1) * 1000)

        # 3. Move to device
        t1 = time.perf_counter()
        tensor = tensor.to(device)
        orig_size[0, 0] = w
        orig_size[0, 1] = h
        t2 = time.perf_counter()
        timings["to_device"].append((t2 - t1) * 1000)

        # 4. Actual inference
        t1 = time.perf_counter()
        blob = {
            "images": tensor,
            "orig_target_sizes": orig_size,
        }
        output = engine(blob)
        t2 = time.perf_counter()
        timings["inference"].append((t2 - t1) * 1000)

        # 5. Synchronize (wait for GPU)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        timings["synchronize"].append((t2 - t1) * 1000)

        # Total time
        t_end_total = time.perf_counter()
        timings["total"].append((t_end_total - t_start_total) * 1000)

    # Calculate and print statistics
    print("\n===== PROFILING RESULTS =====")
    print(f"Averaged over {profile_runs} runs after {warmup_runs} warm-up iterations")
    print("-----------------------------")

    for stage, times in timings.items():
        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        print(f"{stage.ljust(16)}: {avg:.2f} ms (min: {min_t:.2f}, max: {max_t:.2f})")

    # Calculate percentages
    total_avg = sum(timings["total"]) / len(timings["total"])
    print("\n----- Percentage Breakdown -----")
    for stage, times in timings.items():
        if stage != "total":
            avg = sum(times) / len(times)
            percentage = (avg / total_avg) * 100
            print(f"{stage.ljust(16)}: {percentage:.1f}%")

    # Try raw engine timing without any PyTorch overhead
    print("\n----- Direct Engine Timing -----")
    # This depends on how your TRTInference is implemented
    try:
        # Try to access the TensorRT context directly if possible
        # This is just a sketch - you'll need to adapt this to how your engine works
        if hasattr(engine, 'context') and hasattr(engine, 'bindings_addr'):
            # Get some input data ready
            tensor = _tf(pil_img).unsqueeze(0).to(device)
            orig_size[0, 0] = w
            orig_size[0, 1] = h

            # Update input bindings
            blob = {
                "images": tensor,
                "orig_target_sizes": orig_size,
            }

            for n in engine.input_names:
                if blob[n].dtype is not engine.bindings[n].data.dtype:
                    blob[n] = blob[n].to(dtype=engine.bindings[n].data.dtype)
                if engine.bindings[n].shape != blob[n].shape:
                    engine.context.set_input_shape(n, blob[n].shape)
                    engine.bindings[n] = engine.bindings[n]._replace(shape=blob[n].shape)

            engine.bindings_addr.update({n: blob[n].data_ptr() for n in engine.input_names})

            # Time just the execute_v2 call
            torch.cuda.synchronize()

            raw_times = []
            for _ in range(10):
                t1 = time.perf_counter()
                engine.context.execute_v2(list(engine.bindings_addr.values()))
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                raw_times.append((t2 - t1) * 1000)

            raw_avg = sum(raw_times) / len(raw_times)
            print(f"Raw engine execution: {raw_avg:.2f} ms")
    except Exception as e:
        print(f"Failed to do direct engine timing: {e}")


# Add instructions for using the profiler
print("""
To use this profiler:

1. Import this module where you have access to your TensorRT engine
2. Call profile_inference with your engine and a test image path:

   # Example:
   profile_inference(ENGINE, "test_image.jpg", device="cuda:0")

This will provide a detailed breakdown of where time is being spent during inference.
""")