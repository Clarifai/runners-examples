#!/usr/bin/env python3
"""Convert ONNX model to TensorRT engine."""

import argparse
import os

# Initialize CUDA context via PyTorch before importing TensorRT
import torch
_ = torch.cuda.is_available()
if torch.cuda.is_available():
    _ = torch.zeros(1).cuda()

import tensorrt as trt


def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 8,
    workspace_gb: float = 4.0,
    verbose: bool = False,
):
    """Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to input ONNX model
        engine_path: Path to output TensorRT engine
        fp16: Enable FP16 precision
        min_batch: Minimum batch size for dynamic shapes
        opt_batch: Optimal batch size for dynamic shapes
        max_batch: Maximum batch size for dynamic shapes
        workspace_gb: Workspace size in GB
        verbose: Enable verbose logging
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    print(f"Loading ONNX model from {onnx_path}...")

    # Create builder
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    print("ONNX model parsed successfully")

    # Configure builder
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: FP16 not supported on this platform, using FP32")

    # Set up dynamic shape optimization profile
    profile = builder.create_optimization_profile()

    # Get input tensor info
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape

    # Shape format: (batch, channels, height, width)
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    print(f"Input: {input_name}, shape: {input_shape}")

    # Set min/opt/max shapes for dynamic batch
    min_shape = (min_batch, channels, height, width)
    opt_shape = (opt_batch, channels, height, width)
    max_shape = (max_batch, channels, height, width)

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"Dynamic batch configuration:")
    print(f"  Min: {min_shape}")
    print(f"  Opt: {opt_shape}")
    print(f"  Max: {max_shape}")

    # Build engine
    print("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")
    print(f"Engine size: {os.path.getsize(engine_path) / (1024 * 1024):.1f} MB")

    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")
    parser.add_argument(
        "--onnx", "-i",
        type=str,
        required=True,
        help="Path to input ONNX model"
    )
    parser.add_argument(
        "--engine", "-o",
        type=str,
        default="dfine.engine",
        help="Path to output TensorRT engine (default: dfine.engine)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision (default: True)"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision instead of FP16"
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Minimum batch size (default: 1)"
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=1,
        help="Optimal batch size (default: 1)"
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Maximum batch size (default: 8)"
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Workspace size in GB (default: 4.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Handle FP16/FP32 flags
    use_fp16 = not args.fp32

    convert_to_tensorrt(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=use_fp16,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gb=args.workspace,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
