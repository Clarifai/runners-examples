#!/usr/bin/env python3
"""Benchmark script for D-Fine object detection - Local Stream-Stream."""

import sys
sys.path.insert(0, '/home/ubuntu/arman/runners-examples/image-detector/dfine/1')

import time
import io
from PIL import Image as PILImage
import numpy as np
from clarifai.runners.utils.data_types import Image
from model import MyRunner

# Configuration
NUM_REQUESTS = 100
IMAGE_SIZE = (640, 640)


def generate_test_image(size=(640, 640)):
    """Generate a random test image with some shapes."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)

    # Add some colored rectangles to make it more interesting for detection
    # Rectangle 1
    img_array[100:300, 100:300] = [255, 0, 0]  # Red
    # Rectangle 2
    img_array[400:600, 400:700] = [0, 255, 0]  # Green
    # Rectangle 3
    img_array[700:900, 200:500] = [0, 0, 255]  # Blue

    return PILImage.fromarray(img_array)


def image_to_bytes(img):
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def image_generator(images):
    """Generator function for streaming images."""
    for img in images:
        yield img


def run_benchmark():
    """Run benchmark with local stream-stream calls."""
    print(f"=== D-Fine Object Detection Benchmark ===")
    print(f"Mode: LOCAL Stream-Stream")
    print(f"Number of images: {NUM_REQUESTS}")
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print("=" * 50)

    # Generate test images
    print("\nGenerating test images...")
    test_images = []
    for i in range(NUM_REQUESTS):
        img = generate_test_image(IMAGE_SIZE)
        img_bytes = image_to_bytes(img)
        test_images.append(Image(bytes=img_bytes))
    print(f"Generated {NUM_REQUESTS} test images")

    # Initialize model
    print("\nInitializing model...")
    runner = MyRunner()
    runner.load_model()
    print("Model initialized")

    # Warmup
    print("\nRunning warmup...")
    try:
        for _ in runner.stream_image(iter([test_images[0]])):
            pass
        print("Warmup complete")
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run benchmark
    print(f"\nRunning stream benchmark with {NUM_REQUESTS} images...")
    print("-" * 50)

    times = []
    total_start = time.time()

    try:
        # Stream all images and process results
        for i, regions in enumerate(runner.stream_image(image_generator(test_images))):
            end = time.time()
            if i == 0:
                first_start = end

            # For streaming, we measure time from start of stream to each result
            elapsed = end - total_start
            times.append(elapsed)

            num_detections = len(regions)
            print(f"Image {i+1}/{NUM_REQUESTS}: received at {elapsed*1000:.2f}ms ({num_detections} detections)")
    except Exception as e:
        print(f"Stream processing failed: {e}")
        import traceback
        traceback.print_exc()

    total_end = time.time()
    total_time = total_end - total_start

    # Calculate statistics
    print("-" * 50)
    print("\n=== Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful images: {len(times)}/{NUM_REQUESTS}")

    if times:
        # Calculate inter-arrival times (time between consecutive results)
        inter_arrival = []
        for i in range(1, len(times)):
            inter_arrival.append(times[i] - times[i-1])

        if inter_arrival:
            avg_inter = np.mean(inter_arrival)
            min_inter = np.min(inter_arrival)
            max_inter = np.max(inter_arrival)
            std_inter = np.std(inter_arrival)

            fps = 1.0 / avg_inter if avg_inter > 0 else 0
        else:
            avg_inter = times[0]
            min_inter = max_inter = std_inter = 0
            fps = 0

        throughput = len(times) / total_time

        print(f"\nInter-arrival Time Statistics:")
        print(f"  Average: {avg_inter*1000:.2f}ms")
        print(f"  Min: {min_inter*1000:.2f}ms")
        print(f"  Max: {max_inter*1000:.2f}ms")
        print(f"  Std Dev: {std_inter*1000:.2f}ms")

        print(f"\nThroughput:")
        print(f"  FPS (based on avg inter-arrival): {fps:.2f}")
        print(f"  Overall throughput: {throughput:.2f} images/s")
        print(f"  Time to first result: {times[0]*1000:.2f}ms")
        print(f"  Time to last result: {times[-1]*1000:.2f}ms")
    else:
        print("No successful images!")

    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()
