#!/usr/bin/env python3
"""Test the impact of NMS on D-Fine performance."""

import sys
sys.path.insert(0, '/home/ubuntu/arman/runners-examples/image-detector/dfine/1')

import time
import io
from PIL import Image as PILImage
import numpy as np
from clarifai.runners.utils.data_types import Image
from model import MyRunner

NUM_REQUESTS = 50
IMAGE_SIZE = (640, 640)


def generate_test_image(size=(640, 640)):
    """Generate a test image."""
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


def run_test():
    """Test with and without NMS."""
    print(f"=== D-Fine NMS Impact Test ===")
    print(f"Number of requests: {NUM_REQUESTS}")
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print("=" * 60)

    # Generate test images
    print("\nGenerating test images...")
    test_images = []
    for i in range(NUM_REQUESTS):
        img = generate_test_image(IMAGE_SIZE)
        test_images.append(Image(bytes=image_to_bytes(img)))
    print(f"Generated {NUM_REQUESTS} test images")

    # Initialize model
    print("\nInitializing model...")
    runner = MyRunner()
    runner.load_model()
    print("Model initialized\n")

    # Test WITH NMS
    print("=" * 60)
    print("TEST 1: WITH NMS (use_nms=True)")
    print("-" * 60)

    _ = runner.predict(test_images[0], use_nms=True)  # Warmup

    start = time.time()
    all_detections_with_nms = []
    for img in test_images:
        regions = runner.predict(img, use_nms=True)
        all_detections_with_nms.append(len(regions))
    end = time.time()

    time_with_nms = end - start
    fps_with_nms = NUM_REQUESTS / time_with_nms
    avg_detections_with = np.mean(all_detections_with_nms)

    print(f"Total time: {time_with_nms:.3f}s")
    print(f"FPS: {fps_with_nms:.2f}")
    print(f"Avg detections per image: {avg_detections_with:.1f}")

    # Test WITHOUT NMS
    print("\n" + "=" * 60)
    print("TEST 2: WITHOUT NMS (use_nms=False)")
    print("-" * 60)

    _ = runner.predict(test_images[0], use_nms=False)  # Warmup

    start = time.time()
    all_detections_without_nms = []
    for img in test_images:
        regions = runner.predict(img, use_nms=False)
        all_detections_without_nms.append(len(regions))
    end = time.time()

    time_without_nms = end - start
    fps_without_nms = NUM_REQUESTS / time_without_nms
    avg_detections_without = np.mean(all_detections_without_nms)

    print(f"Total time: {time_without_nms:.3f}s")
    print(f"FPS: {fps_without_nms:.2f}")
    print(f"Avg detections per image: {avg_detections_without:.1f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    speedup = fps_without_nms / fps_with_nms
    time_saved = time_with_nms - time_without_nms
    time_saved_pct = (time_saved / time_with_nms) * 100

    detection_diff = avg_detections_without - avg_detections_with
    detection_diff_pct = (detection_diff / avg_detections_with * 100) if avg_detections_with > 0 else 0

    print(f"\nWith NMS:    {fps_with_nms:.2f} FPS, {avg_detections_with:.1f} detections/image")
    print(f"Without NMS: {fps_without_nms:.2f} FPS, {avg_detections_without:.1f} detections/image")
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time saved: {time_saved:.3f}s ({time_saved_pct:.1f}%)")
    print(f"Detection difference: {detection_diff:+.1f} ({detection_diff_pct:+.1f}%)")

    if abs(detection_diff_pct) < 5:
        print("\n✓ Detection counts nearly identical - D-FINE doesn't need NMS!")
        print(f"  Recommendation: Set use_nms=False (default) for {speedup:.2f}x speedup")
    elif detection_diff > 0:
        print(f"\n⚠️  Without NMS produces {detection_diff:.1f} more detections")
        print("  D-FINE might benefit from NMS to remove duplicates")
    else:
        print(f"\n✓ NMS removes {-detection_diff:.1f} redundant detections")
        print(f"  But costs {time_saved_pct:.1f}% performance")

    print("=" * 60)


if __name__ == "__main__":
    run_test()
