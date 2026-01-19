#!/usr/bin/env python3
"""Fair comparison between unary and stream performance - measuring individual calls."""

import sys
sys.path.insert(0, '/home/ubuntu/arman/runners-examples/image-detector/dfine/1')

import time
import io
from PIL import Image as PILImage
import numpy as np
from clarifai.runners.utils.data_types import Image
from model import MyRunner

# Configuration
NUM_REQUESTS = 10
IMAGE_SIZE = (1024, 1024)


def generate_test_image(size=(1024, 1024)):
    """Generate a random test image with some shapes."""
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img_array[100:300, 100:300] = [255, 0, 0]  # Red
    img_array[400:600, 400:700] = [0, 255, 0]  # Green
    img_array[700:900, 200:500] = [0, 0, 255]  # Blue
    return PILImage.fromarray(img_array)


def image_to_bytes(img):
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def run_comparison():
    """Compare unary vs stream with fair timing."""
    print(f"=== Fair Comparison: Unary vs Stream ===")
    print(f"Number of requests: {NUM_REQUESTS}")
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
    print("\nWarming up...")
    _ = runner.predict(test_images[0])
    for _ in runner.stream_image(iter([test_images[0]])):
        pass
    print("Warmup complete\n")

    # Test 1: Unary calls
    print("=" * 50)
    print("TEST 1: Unary calls (individual predict())")
    print("-" * 50)

    unary_times = []
    for i in range(NUM_REQUESTS):
        start = time.time()
        regions = runner.predict(test_images[i])
        end = time.time()
        elapsed = end - start
        unary_times.append(elapsed)
        print(f"Request {i+1}: {elapsed*1000:.2f}ms ({len(regions)} detections)")

    # Test 2: Stream calls - measure each iteration separately
    print("\n" + "=" * 50)
    print("TEST 2: Stream calls (stream_image with per-iteration timing)")
    print("-" * 50)

    stream_times = []
    last_time = time.time()
    for i, regions in enumerate(runner.stream_image(iter(test_images))):
        current_time = time.time()
        elapsed = current_time - last_time
        stream_times.append(elapsed)
        print(f"Request {i+1}: {elapsed*1000:.2f}ms ({len(regions)} detections)")
        last_time = current_time

    # Calculate statistics
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    unary_avg = np.mean(unary_times) * 1000
    unary_fps = 1.0 / np.mean(unary_times)

    stream_avg = np.mean(stream_times) * 1000
    stream_fps = 1.0 / np.mean(stream_times)

    print(f"\nUnary calls:")
    print(f"  Avg latency: {unary_avg:.2f}ms")
    print(f"  Min: {np.min(unary_times)*1000:.2f}ms")
    print(f"  Max: {np.max(unary_times)*1000:.2f}ms")
    print(f"  FPS: {unary_fps:.2f}")

    print(f"\nStream calls:")
    print(f"  Avg latency: {stream_avg:.2f}ms")
    print(f"  Min: {np.min(stream_times)*1000:.2f}ms")
    print(f"  Max: {np.max(stream_times)*1000:.2f}ms")
    print(f"  FPS: {stream_fps:.2f}")

    overhead = ((stream_avg - unary_avg) / unary_avg) * 100
    print(f"\nStream overhead: {overhead:+.1f}%")
    if abs(overhead) < 5:
        print("✓ Performance is essentially identical!")
    elif overhead > 0:
        print(f"⚠ Stream is {overhead:.1f}% slower (generator/iteration overhead)")
    else:
        print(f"✓ Stream is {-overhead:.1f}% faster (unexpected but good!)")

    print("=" * 50)


if __name__ == "__main__":
    run_comparison()
