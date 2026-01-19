#!/bin/bash
# Build TensorRT engine from D-FINE model checkpoint
# Usage: ./build_tensorrt.sh <checkpoint_path> [output_dir]

set -e

# Default values
CHECKPOINT_PATH="${1:-checkpoints}"
OUTPUT_DIR="${2:-.}"
ONNX_FILE="${OUTPUT_DIR}/dfine.onnx"
ENGINE_FILE="${OUTPUT_DIR}/dfine.engine"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "D-FINE TensorRT Build Pipeline"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================"

# Step 1: Export to ONNX
echo ""
echo "Step 1/2: Exporting to ONNX..."
echo "----------------------------------------"
python "${SCRIPT_DIR}/export_onnx.py" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output "${ONNX_FILE}" \
    --opset 17

# Check ONNX export succeeded
if [ ! -f "${ONNX_FILE}" ]; then
    echo "Error: ONNX export failed!"
    exit 1
fi

echo ""
echo "Step 2/2: Converting to TensorRT..."
echo "----------------------------------------"
python "${SCRIPT_DIR}/convert_tensorrt.py" \
    --onnx "${ONNX_FILE}" \
    --engine "${ENGINE_FILE}" \
    --fp16 \
    --min-batch 1 \
    --opt-batch 1 \
    --max-batch 8 \
    --workspace 4.0

# Check TensorRT conversion succeeded
if [ ! -f "${ENGINE_FILE}" ]; then
    echo "Error: TensorRT conversion failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "Build Complete!"
echo "============================================"
echo "ONNX model:      ${ONNX_FILE}"
echo "TensorRT engine: ${ENGINE_FILE}"
echo ""
echo "To use the TensorRT engine, place it in:"
echo "  <model_directory>/dfine.engine"
echo "============================================"
