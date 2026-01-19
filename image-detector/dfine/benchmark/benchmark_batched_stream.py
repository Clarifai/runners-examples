#!/usr/bin/env python3
"""Benchmark batched streaming vs unary to show streaming performance gains."""

import sys
sys.path.insert(0, '/home/ubuntu/arman/runners-examples/image-detector/dfine/1')

import time
import io
from PIL import Image as PILImage
import numpy as np
from clarifai.runners.utils.data_types import Image
from model import MyRunner

# Configuration
NUM_REQUESTS = 20  # More requests to see batching benefits
IMAGE_SIZE = (1024, 1024)
BATCH_SIZES = [1, 2, 4, 8]


def generate_test_image(size=(1024, 1024)):
    """Generate a random test image with some shapes."""
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
    """Compare unary vs batched streaming with different batch sizes."""
    print(f"=== Batched Streaming Performance Test ===")
    print(f"Number of images: {NUM_REQUESTS}")
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Testing batch sizes: {BATCH_SIZES}")
    print("=" * 60)

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
    for _ in runner.stream_image(iter([test_images[0]]), batch_size=1):
        pass
    print("Warmup complete\n")

    results = {}

    # Test 1: Unary (baseline)
    print("=" * 60)
    print("TEST 1: Unary calls (baseline)")
    print("-" * 60)

    start = time.time()
    for i in range(NUM_REQUESTS):
        regions = runner.predict(test_images[i])
    end = time.time()

    unary_time = end - start
    unary_fps = NUM_REQUESTS / unary_time
    results['unary'] = {'time': unary_time, 'fps': unary_fps}

    print(f"Total time: {unary_time:.3f}s")
    print(f"FPS: {unary_fps:.2f}")
    print(f"Avg per image: {(unary_time/NUM_REQUESTS)*1000:.2f}ms")

    # Test 2-N: Different batch sizes
    for batch_size in BATCH_SIZES:
        print("\n" + "=" * 60)
        print(f"TEST: Batched streaming (batch_size={batch_size})")
        print("-" * 60)

        start = time.time()
        count = 0
        for regions in runner.stream_image(iter(test_images), batch_size=batch_size):
            count += 1
        end = time.time()

        batch_time = end - start
        batch_fps = NUM_REQUESTS / batch_time
        speedup = batch_fps / unary_fps
        results[f'batch_{batch_size}'] = {
            'time': batch_time,
            'fps': batch_fps,
            'speedup': speedup
        }

        print(f"Total time: {batch_time:.3f}s")
        print(f"FPS: {batch_fps:.2f}")
        print(f"Avg per image: {(batch_time/NUM_REQUESTS)*1000:.2f}ms")
        print(f"Speedup vs unary: {speedup:.2f}x")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<20} {'Time (s)':<12} {'FPS':<12} {'Speedup':<12}")
    print("-" * 60)

    print(f"{'Unary':<20} {results['unary']['time']:<12.3f} {results['unary']['fps']:<12.2f} {'1.00x':<12}")

    for batch_size in BATCH_SIZES:
        key = f'batch_{batch_size}'
        if key in results:
            r = results[key]
            print(f"{'Batch ' + str(batch_size):<20} {r['time']:<12.3f} {r['fps']:<12.2f} {r['speedup']:<12.2f}x")

    # Find best
    best_key = max(results.keys(), key=lambda k: results[k]['fps'])
    best = results[best_key]

    print("\n" + "=" * 60)
    if best_key == 'unary':
        print("⚠️  Unary is still fastest - batching didn't help!")
        print("This could mean:")
        print("  - Batch size too small")
        print("  - GPU not fully utilized even with single images")
        print("  - Batching overhead outweighs benefits")
    else:
        batch_num = best_key.split('_')[1]
        print(f"✓ BEST: Batch size {batch_num}")
        print(f"  FPS: {best['fps']:.2f}")
        print(f"  Speedup: {best['speedup']:.2f}x faster than unary")
        print(f"  Time saved: {(results['unary']['time'] - best['time']):.3f}s ({((results['unary']['time'] - best['time'])/results['unary']['time']*100):.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
