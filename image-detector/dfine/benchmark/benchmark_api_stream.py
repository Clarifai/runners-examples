#!/usr/bin/env python3
"""Benchmark script for D-Fine object detection - API Stream-Stream requests."""

import time
import io
import os
from PIL import Image as PILImage
import numpy as np
from clarifai.client.model import Model
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# Configuration
NUM_REQUESTS = 100
IMAGE_SIZE = (640, 640)
MODEL_URL = "https://clarifai.com/arman/local-runner-app/models/local-runner-model"

# Get PAT from environment
PAT = os.environ.get("CLARIFAI_PAT", "")


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


def create_input_iterator(image_bytes_list):
    """Create an iterator that yields PostModelOutputsRequest for streaming."""
    for img_bytes in image_bytes_list:
        input_proto = resources_pb2.Input(
            data=resources_pb2.Data(
                image=resources_pb2.Image(base64=img_bytes)
            )
        )
        yield service_pb2.PostModelOutputsRequest(
            inputs=[input_proto]
        )


def run_benchmark():
    """Run benchmark with API stream-stream requests."""
    if not PAT:
        print("ERROR: CLARIFAI_PAT environment variable not set!")
        print("Please set it with: export CLARIFAI_PAT='your_pat_here'")
        return

    print(f"=== D-Fine Object Detection Benchmark ===")
    print(f"Mode: API Stream-Stream")
    print(f"Number of images: {NUM_REQUESTS}")
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

    # Run benchmark with streaming
    print(f"\nRunning stream benchmark with {NUM_REQUESTS} images...")
    print("-" * 50)

    times = []
    total_start = time.time()

    try:
        # Use stream_inputs method if available, otherwise fall back to sequential calls
        if hasattr(model, 'stream_inputs'):
            print("Using stream_inputs method...")
            for i, output in enumerate(model.stream_inputs(test_images, input_type="image")):
                end = time.time()
                elapsed = end - total_start
                times.append(elapsed)

                num_detections = len(output.data.regions) if hasattr(output.data, 'regions') else 0
                print(f"Image {i+1}/{NUM_REQUESTS}: received at {elapsed*1000:.2f}ms ({num_detections} detections)")
        else:
            print("Stream API not available, using sequential API calls...")
            # Fallback: simulate streaming with rapid sequential calls
            for i, img_bytes in enumerate(test_images):
                try:
                    response = model.predict_by_bytes(img_bytes, input_type="image")
                    end = time.time()
                    elapsed = end - total_start
                    times.append(elapsed)

                    num_detections = len(response.outputs[0].data.regions) if response.outputs else 0
                    print(f"Image {i+1}/{NUM_REQUESTS}: received at {elapsed*1000:.2f}ms ({num_detections} detections)")
                except Exception as e:
                    print(f"Image {i+1}/{NUM_REQUESTS}: FAILED - {e}")
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
        # Calculate inter-arrival times
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
