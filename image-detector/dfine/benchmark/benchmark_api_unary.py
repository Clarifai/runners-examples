#!/usr/bin/env python3
"""Benchmark script for D-Fine object detection - API Unary-Unary requests."""

import time
import io
import os
from PIL import Image as PILImage
import numpy as np
from clarifai.client.model import Model

# Configuration
NUM_REQUESTS = 100
IMAGE_SIZE = (1024, 1024)
MODEL_URL = "https://clarifai.com/arman/local-runner-app/models/local-runner-model"

# Get PAT from environment or use the one from user's example
PAT = os.environ.get("CLARIFAI_PAT", "")


def generate_test_image(size=(1024, 1024)):
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


def run_benchmark():
    """Run benchmark with API unary-unary requests."""
    if not PAT:
        print("ERROR: CLARIFAI_PAT environment variable not set!")
        print("Please set it with: export CLARIFAI_PAT='your_pat_here'")
        return

    print(f"=== D-Fine Object Detection Benchmark ===")
    print(f"Mode: API Unary-Unary (sequential requests)")
    print(f"Number of requests: {NUM_REQUESTS}")
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Model URL: {MODEL_URL}")
    print("=" * 50)

    # Generate test images
    print("\nGenerating test images...")
    test_images = []
    for i in range(NUM_REQUESTS):
        img = generate_test_image(IMAGE_SIZE)
        test_images.append(image_to_bytes(img))
    print(f"Generated {NUM_REQUESTS} test images")

    # Initialize model
    print("\nInitializing model connection...")
    try:
        model = Model(url=MODEL_URL, pat=PAT)
        print("Model connection established")
    except Exception as e:
        print(f"Failed to connect to model: {e}")
        return

    # Warmup request
    print("\nRunning warmup request...")
    try:
        _ = model.predict_by_bytes(test_images[0], input_type="image")
        print("Warmup complete")
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run benchmark
    print(f"\nRunning benchmark with {NUM_REQUESTS} requests...")
    print("-" * 50)

    times = []
    total_start = time.time()

    for i in range(NUM_REQUESTS):
        start = time.time()
        try:
            response = model.predict_by_bytes(test_images[i], input_type="image")
            end = time.time()
            elapsed = end - start
            times.append(elapsed)

            # Count detections
            num_detections = len(response.outputs[0].data.regions) if response.outputs else 0

            print(f"Request {i+1}/{NUM_REQUESTS}: {elapsed*1000:.2f}ms ({num_detections} detections)")
        except Exception as e:
            print(f"Request {i+1}/{NUM_REQUESTS}: FAILED - {e}")

    total_end = time.time()
    total_time = total_end - total_start

    # Calculate statistics
    print("-" * 50)
    print("\n=== Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful requests: {len(times)}/{NUM_REQUESTS}")

    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        fps = 1.0 / avg_time if avg_time > 0 else 0
        throughput = len(times) / total_time

        print(f"\nLatency Statistics:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Min: {min_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")
        print(f"  Std Dev: {std_time*1000:.2f}ms")

        print(f"\nThroughput:")
        print(f"  FPS (avg per request): {fps:.2f}")
        print(f"  Overall throughput: {throughput:.2f} req/s")
    else:
        print("No successful requests!")

    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()
