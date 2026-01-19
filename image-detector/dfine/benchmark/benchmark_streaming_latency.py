#!/usr/bin/env python3
"""Benchmark streaming latency vs throughput tradeoff with different batch sizes."""

import sys
sys.path.insert(0, '/home/ubuntu/arman/runners-examples/image-detector/dfine/1')

import time
import io
from PIL import Image as PILImage
import numpy as np
from clarifai.runners.utils.data_types import Image
from model import MyRunner

# Configuration
NUM_FRAMES = 30  # Simulate 30 frames
IMAGE_SIZE = (1024, 1024)
BATCH_SIZES = [1, 2, 4, 8]


def generate_test_image(size=(1024, 1024)):
    """Generate a random test image."""
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img_array[100:300, 100:300] = [255, 0, 0]
    img_array[400:600, 400:700] = [0, 255, 0]
    img_array[700:900, 200:500] = [0, 0, 255]
    return PILImage.fromarray(img_array)


def image_to_bytes(img):
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def run_benchmark():
    """Measure both throughput and per-frame latency."""
    print(f"=== Streaming Latency vs Throughput Analysis ===")
    print(f"Frames: {NUM_FRAMES}")
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Simulating real-time video streaming scenario")
    print("=" * 70)

    # Generate test images
    print("\nGenerating test frames...")
    test_images = []
    for i in range(NUM_FRAMES):
        img = generate_test_image(IMAGE_SIZE)
        img_bytes = image_to_bytes(img)
        test_images.append(Image(bytes=img_bytes))
    print(f"Generated {NUM_FRAMES} frames")

    # Initialize model
    print("\nInitializing model...")
    runner = MyRunner()
    runner.load_model()
    print("Model initialized")

    # Warmup
    print("\nWarming up...")
    _ = runner.predict(test_images[0])
    print("Warmup complete\n")

    results = {}

    for batch_size in BATCH_SIZES:
        print("=" * 70)
        print(f"BATCH SIZE: {batch_size}")
        print("-" * 70)

        frame_latencies = []
        total_start = time.time()

        # Track when each frame enters processing
        frame_enter_times = []
        frame_idx = 0

        for regions in runner.stream_image(iter(test_images), batch_size=batch_size):
            result_time = time.time()

            # For batched processing, the latency is measured from when the FIRST
            # frame in the batch entered until results come out
            if frame_idx % batch_size == 0:
                # Starting a new batch
                batch_start_time = result_time

            # Calculate per-frame latency
            # In real streaming, each frame waits from when it arrives until result
            frame_position_in_batch = frame_idx % batch_size

            # Estimate when this frame "arrived" (entered the batch)
            # First frame waits longest, last frame waits shortest
            if frame_idx < len(test_images):
                # Simulate: frames arrive, get batched, then processed
                # The latency for frame N in a batch is the time from when frame N arrived
                # to when the batch completes
                latency = (batch_size - frame_position_in_batch) * 0.001  # Simulated arrival offset
                frame_latencies.append(latency)

            frame_idx += 1

        total_end = time.time()
        total_time = total_end - total_start
        fps = NUM_FRAMES / total_time
        avg_latency = (total_time / NUM_FRAMES) * 1000

        # Calculate actual latency impact
        # In batching, first frame in batch waits for (batch_size-1) other frames
        max_batch_wait = (batch_size - 1) * avg_latency

        results[batch_size] = {
            'total_time': total_time,
            'fps': fps,
            'avg_latency': avg_latency,
            'max_batch_wait': max_batch_wait
        }

        print(f"Total time: {total_time:.3f}s")
        print(f"Throughput: {fps:.2f} FPS")
        print(f"Avg time per frame: {avg_latency:.2f}ms")
        print(f"Max batching wait (first frame): ~{max_batch_wait:.2f}ms")

        if batch_size > 1:
            print(f"⚠️  Latency penalty: First frame in batch waits {max_batch_wait:.0f}ms")
            print(f"   (waiting for {batch_size-1} more frames to fill batch)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Latency vs Throughput Tradeoff")
    print("=" * 70)
    print(f"{'Batch':<8} {'FPS':<12} {'Throughput':<15} {'Max Latency':<20} {'Use Case':<20}")
    print("-" * 70)

    for batch_size in BATCH_SIZES:
        r = results[batch_size]
        speedup = r['fps'] / results[1]['fps']

        if batch_size == 1:
            use_case = "Real-time streaming"
            indicator = "✓ LOW LATENCY"
        elif batch_size <= 4:
            use_case = "Balanced"
            indicator = "⚖️ MEDIUM"
        else:
            use_case = "Offline/batch"
            indicator = "⚠️ HIGH LATENCY"

        print(f"{batch_size:<8} {r['fps']:<12.2f} "
              f"{speedup:.2f}x{'':<10} "
              f"~{r['max_batch_wait']:<18.0f}ms "
              f"{indicator}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("📹 Real-time streaming (live video):  batch_size=1  (lowest latency)")
    print("⚡ Balanced (near real-time):         batch_size=2  (moderate latency)")
    print("🚀 Maximum throughput (offline):      batch_size=4  (high latency)")
    print("\nFor streaming, latency matters more than throughput!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
