#!/usr/bin/env python3
"""Export D-FINE model to ONNX format for TensorRT conversion."""

import argparse
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from transformers import DFineForObjectDetection


class DFineONNXWrapper(nn.Module):
    """Wrapper to export only the necessary outputs from D-FINE model."""

    def __init__(self, model: DFineForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple:
        """Forward pass returning only logits and pred_boxes."""
        outputs = self.model(pixel_values=pixel_values)
        # Return logits and pred_boxes (normalized [cx, cy, w, h] format)
        return outputs.logits, outputs.pred_boxes


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    simplify_model: bool = True,
    dynamic_batch: bool = True,
):
    """Export D-FINE model to ONNX format.

    Args:
        checkpoint_path: Path to D-FINE checkpoint
        output_path: Output ONNX file path
        opset_version: ONNX opset version (17 recommended for D-FINE)
        simplify_model: Whether to simplify the ONNX model
        dynamic_batch: Whether to support dynamic batch size
    """
    from transformers import AutoImageProcessor

    print(f"Loading model from {checkpoint_path}...")
    model = DFineForObjectDetection.from_pretrained(checkpoint_path)
    model.eval()

    # Get input dimensions from processor
    processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    if isinstance(processor.size, dict):
        height = processor.size.get("height", 640)
        width = processor.size.get("width", 640)
    else:
        height = width = processor.size

    print(f"Model loaded. Input size: {height}x{width}")

    # Create wrapper for clean export
    wrapper = DFineONNXWrapper(model)
    wrapper.eval()

    # Create dummy input (batch_size=1, channels=3, height, width)
    dummy_input = torch.randn(1, 3, height, width)

    # Set up dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        }

    print(f"Exporting to ONNX with opset {opset_version}...")

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
    )

    print(f"ONNX model exported to {output_path}")

    # Verify the model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Simplify the model
    if simplify_model:
        print("Simplifying ONNX model...")
        simplified_model, check = simplify(onnx_model)
        if check:
            onnx.save(simplified_model, output_path)
            print("Model simplified successfully!")
        else:
            print("Warning: Simplification check failed, keeping original model")

    # Print model info
    onnx_model = onnx.load(output_path)
    print("\nModel inputs:")
    for inp in onnx_model.graph.input:
        print(f"  {inp.name}: {[d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]}")

    print("\nModel outputs:")
    for out in onnx_model.graph.output:
        print(f"  {out.name}: {[d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]}")

    print(f"\nExport complete: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export D-FINE to ONNX format")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to D-FINE checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dfine.onnx",
        help="Output ONNX file path (default: dfine.onnx)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, required for D-FINE accuracy)"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification"
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Use static batch size instead of dynamic"
    )

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        simplify_model=not args.no_simplify,
        dynamic_batch=not args.static_batch,
    )


if __name__ == "__main__":
    main()
