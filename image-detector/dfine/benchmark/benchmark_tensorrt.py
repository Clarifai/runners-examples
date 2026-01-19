#!/usr/bin/env python3
"""Benchmark comparing PyTorch vs TensorRT inference for D-FINE model."""

import argparse
import os
import sys
import time
from typing import Optional

import torch
import numpy as np
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TensorRT imports (optional)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None


class TensorRTInference:
    """TensorRT inference engine wrapper for D-FINE model."""

    def __init__(self, engine_path: str):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")

        self.logger = trt.Logger(trt.Logger.WARNING)

        print(f"Loading TensorRT engine from {engine_path}...")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(1, self.engine.num_io_tensors)
        ]

        input_shape = self.engine.get_tensor_shape(self.input_name)
        print(f"TensorRT engine loaded: input shape = {input_shape}")

    def __call__(self, pixel_values: torch.Tensor):
        batch_size = pixel_values.shape[0]

        if not pixel_values.is_cuda:
            pixel_values = pixel_values.cuda()
        pixel_values = pixel_values.contiguous()

        self.context.set_input_shape(self.input_name, pixel_values.shape)

        outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            shape[0] = batch_size
            outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda").contiguous()

        self.context.set_tensor_address(self.input_name, pixel_values.data_ptr())
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return outputs.get("logits"), outputs.get("pred_boxes")


def benchmark_pytorch(
    model: DFineForObjectDetection,
    processor: AutoImageProcessor,
    image: Image.Image,
    warmup_iterations: int = 50,
    benchmark_iterations: int = 1000,
) -> dict:
    """Benchmark PyTorch model inference."""
    device = next(model.parameters()).device

    # Prepare input
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    print(f"Running PyTorch benchmark ({warmup_iterations} warmup, {benchmark_iterations} iterations)...")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(pixel_values=pixel_values)

    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            _ = model(pixel_values=pixel_values)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        "backend": "PyTorch",
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "fps": 1000 / np.mean(latencies),
    }


def benchmark_tensorrt(
    trt_engine: TensorRTInference,
    processor: AutoImageProcessor,
    image: Image.Image,
    warmup_iterations: int = 50,
    benchmark_iterations: int = 1000,
) -> dict:
    """Benchmark TensorRT engine inference."""
    # Prepare input
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda()

    print(f"Running TensorRT benchmark ({warmup_iterations} warmup, {benchmark_iterations} iterations)...")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = trt_engine(pixel_values)

    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            _ = trt_engine(pixel_values)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        "backend": "TensorRT FP16",
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "fps": 1000 / np.mean(latencies),
    }


def verify_outputs(
    pytorch_model: DFineForObjectDetection,
    trt_engine: TensorRTInference,
    processor: AutoImageProcessor,
    image: Image.Image,
    atol: float = 1e-2,
) -> bool:
    """Verify TensorRT outputs match PyTorch outputs."""
    device = next(pytorch_model.parameters()).device

    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        # PyTorch inference
        pytorch_output = pytorch_model(pixel_values=pixel_values)
        pytorch_logits = pytorch_output.logits
        pytorch_boxes = pytorch_output.pred_boxes

        # TensorRT inference
        trt_logits, trt_boxes = trt_engine(pixel_values)

    # Compare outputs
    logits_match = torch.allclose(pytorch_logits, trt_logits, atol=atol, rtol=1e-3)
    boxes_match = torch.allclose(pytorch_boxes, trt_boxes, atol=atol, rtol=1e-3)

    logits_diff = (pytorch_logits - trt_logits).abs().max().item()
    boxes_diff = (pytorch_boxes - trt_boxes).abs().max().item()

    print(f"\nOutput verification (atol={atol}):")
    print(f"  Logits match: {logits_match} (max diff: {logits_diff:.6f})")
    print(f"  Boxes match: {boxes_match} (max diff: {boxes_diff:.6f})")

    return logits_match and boxes_match


def print_results(results: list):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    headers = ["Backend", "Mean (ms)", "Std (ms)", "P50 (ms)", "P95 (ms)", "FPS"]
    row_format = "{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}"

    print(row_format.format(*headers))
    print("-" * 70)

    for r in results:
        print(row_format.format(
            r["backend"],
            f"{r['mean_latency_ms']:.2f}",
            f"{r['std_latency_ms']:.2f}",
            f"{r['p50_latency_ms']:.2f}",
            f"{r['p95_latency_ms']:.2f}",
            f"{r['fps']:.1f}",
        ))

    print("=" * 70)

    # Calculate speedup
    if len(results) == 2:
        pytorch_fps = results[0]["fps"]
        trt_fps = results[1]["fps"]
        speedup = trt_fps / pytorch_fps
        print(f"\nSpeedup: {speedup:.2f}x ({pytorch_fps:.1f} FPS -> {trt_fps:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT for D-FINE")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to D-FINE checkpoint directory"
    )
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default=None,
        help="Path to TensorRT engine (default: <checkpoint>/dfine.engine)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to test image (default: generate random)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="Number of benchmark iterations (default: 1000)"
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Only benchmark PyTorch model"
    )
    parser.add_argument(
        "--tensorrt-only",
        action="store_true",
        help="Only benchmark TensorRT engine"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip output verification"
    )

    args = parser.parse_args()

    # Set engine path
    engine_path = args.engine or os.path.join(args.checkpoint, "dfine.engine")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Engine: {engine_path}")

    # Load processor
    processor = AutoImageProcessor.from_pretrained(args.checkpoint, use_fast=True)

    # Get input dimensions from processor config
    image_size = processor.size
    if isinstance(image_size, dict):
        height = image_size.get("height", 640)
        width = image_size.get("width", 640)
    else:
        height = width = image_size

    # Load or create test image
    if args.image:
        print(f"Loading test image: {args.image}")
        image = Image.open(args.image).convert("RGB")
    else:
        print(f"Using random test image ({width}x{height})")
        image = Image.fromarray(
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        )

    results = []

    # Benchmark PyTorch
    if not args.tensorrt_only:
        print("\nLoading PyTorch model...")
        pytorch_model = DFineForObjectDetection.from_pretrained(args.checkpoint).cuda()
        pytorch_model.eval()

        pytorch_results = benchmark_pytorch(
            pytorch_model, processor, image,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
        )
        results.append(pytorch_results)

    # Benchmark TensorRT
    if not args.pytorch_only:
        if not TENSORRT_AVAILABLE:
            print("\nWarning: TensorRT not available, skipping TensorRT benchmark")
        elif not os.path.exists(engine_path):
            print(f"\nWarning: TensorRT engine not found at {engine_path}, skipping TensorRT benchmark")
        else:
            print("\nLoading TensorRT engine...")
            trt_engine = TensorRTInference(engine_path)

            trt_results = benchmark_tensorrt(
                trt_engine, processor, image,
                warmup_iterations=args.warmup,
                benchmark_iterations=args.iterations,
            )
            results.append(trt_results)

            # Verify outputs
            if not args.skip_verify and not args.tensorrt_only:
                verify_outputs(pytorch_model, trt_engine, processor, image)

    # Print results
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
