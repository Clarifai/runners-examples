#!/usr/bin/env python3
"""Stable benchmark - runs for 1 min, discards first half."""

import sys
sys.path.insert(0, '/home/ubuntu/arman/runners-examples/image-detector/dfine/1')

import time
import io
from PIL import Image as PILImage
import numpy as np
from clarifai.runners.utils.data_types import Image
from model import MyRunner

# Test native resolution only (D-FINE uses 640x640)
TEST_CONFIGS = [
    {"size": (640, 640), "name": "640x640 (native)"},
]

TARGET_DURATION = 60  # Run for 60 seconds


def generate_test_image(size=(640, 640)):
    """Generate a test image."""
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img_array[50:150, 50:150] = [255, 0, 0]
    img_array[200:350, 200:400] = [0, 255, 0]
    img_array[400:550, 100:300] = [0, 0, 255]
    return PILImage.fromarray(img_array)


def image_to_bytes(img):
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def run_benchmark(image_size, name):
    """Run benchmark for given image size."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    # Generate test images
    print("Generating test images...")
    test_images = []
    for i in range(200):  # Pre-generate enough images
        img = generate_test_image(image_size)
        test_images.append(Image(bytes=image_to_bytes(img)))
    print(f"Generated {len(test_images)} test images")

    # Initialize model
    print("Initializing model...")
    runner = MyRunner()
    runner.load_model()
    print("Model initialized")

    # Run for target duration
    print(f"\nRunning benchmark for {TARGET_DURATION} seconds...")
    print("(First 50% will be discarded as warmup)")
    print("-" * 70)

    all_times = []
    all_detections = []
    start_time = time.time()
    img_idx = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > TARGET_DURATION:
            break

        # Cycle through images
        img = test_images[img_idx % len(test_images)]
        img_idx += 1

        iter_start = time.time()
        regions = runner.predict(img)
        iter_time = time.time() - iter_start

        all_times.append(iter_time)
        all_detections.append(len(regions))

        # Progress update every 5 seconds
        if len(all_times) % 50 == 0:
            print(f"  {elapsed:.1f}s: {len(all_times)} images processed...")

    total_time = time.time() - start_time
    print(f"\nCompleted: {len(all_times)} images in {total_time:.2f}s")

    # Discard first 50%
    discard_count = len(all_times) // 2
    stable_times = all_times[discard_count:]
    stable_detections = all_detections[discard_count:]

    print(f"Discarded first {discard_count} images (warmup)")
    print(f"Analyzing {len(stable_times)} stable images")

    # Calculate statistics
    avg_time = np.mean(stable_times)
    min_time = np.min(stable_times)
    max_time = np.max(stable_times)
    std_time = np.std(stable_times)
    fps = 1.0 / avg_time

    avg_detections = np.mean(stable_detections)

    print("\n" + "=" * 70)
    print("RESULTS (after warmup)")
    print("=" * 70)
    print(f"Images analyzed: {len(stable_times)}")
    print(f"Avg latency: {avg_time*1000:.2f}ms")
    print(f"Min latency: {min_time*1000:.2f}ms")
    print(f"Max latency: {max_time*1000:.2f}ms")
    print(f"Std dev: {std_time*1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
    print(f"Avg detections: {avg_detections:.1f}")
    print("=" * 70)

    return {
        'name': name,
        'size': image_size,
        'fps': fps,
        'avg_latency': avg_time * 1000,
        'min_latency': min_time * 1000,
        'max_latency': max_time * 1000,
        'std_latency': std_time * 1000,
        'images': len(stable_times),
        'detections': avg_detections
    }


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("D-FINE STABLE BENCHMARK")
    print("=" * 70)
    print(f"Duration: {TARGET_DURATION}s per test")
    print("Warmup: First 50% discarded")
    print("=" * 70)

    results = []
    for config in TEST_CONFIGS:
        result = run_benchmark(config['size'], config['name'])
        results.append(result)
        time.sleep(2)  # Cool down between tests

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Resolution':<20} {'FPS':<12} {'Avg Latency':<15} {'Images':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<20} {r['fps']:<12.2f} {r['avg_latency']:<15.2f}ms {r['images']:<12}")

    print("=" * 70)


if __name__ == "__main__":
    main()
