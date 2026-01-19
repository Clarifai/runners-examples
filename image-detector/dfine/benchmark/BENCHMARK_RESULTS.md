# D-Fine Object Detection Benchmark Results

## Test Configuration
- **Image Size**: 640x640 (native D-FINE resolution)
- **Number of Requests**: 100 per test
- **Model**: ustc-community/dfine-small-obj2coco
- **Device**: CUDA (GPU - NVIDIA A10)
- **Backend**: TensorRT FP16 (with PyTorch fallback)

## TensorRT FP16 Results (Current)

### 1. LOCAL Stream-Stream
**Command**: `python benchmark_local_stream.py`

| Metric | Value |
|--------|-------|
| Avg Inter-arrival | 14.00ms |
| Min Inter-arrival | 7.34ms |
| Max Inter-arrival | 27.82ms |
| Std Dev | 7.53ms |
| **FPS** | **71.43** |
| Overall Throughput | 71.56 images/s |
| Time to First Result | 11.38ms |
| Time to Last Result | 1397.39ms |

### 2. LOCAL Unary-Unary (Direct Model Calls)
**Command**: `python benchmark_direct.py`

| Metric | Value |
|--------|-------|
| Average Latency | 12.94ms |
| Min Latency | 7.11ms |
| Max Latency | 35.38ms |
| Std Dev | 7.21ms |
| **FPS** | **77.30** |
| Throughput | 77.25 req/s |

### 3. API Stream-Stream
**Command**: `export CLARIFAI_PAT='your_pat' && python benchmark_api_stream.py`

| Metric | Value |
|--------|-------|
| Avg Inter-arrival | 37.18ms |
| Min Inter-arrival | 31.70ms |
| Max Inter-arrival | 58.32ms |
| Std Dev | 4.06ms |
| **FPS** | **26.90** |
| Overall Throughput | 22.68 images/s |
| Time to First Result | 727.68ms |
| Time to Last Result | 4408.28ms |

### 4. API Unary-Unary
**Command**: `export CLARIFAI_PAT='your_pat' && python benchmark_api_unary.py`

| Metric | Value |
|--------|-------|
| Average Latency | 43.17ms |
| Min Latency | 36.26ms |
| Max Latency | 90.55ms |
| Std Dev | 7.98ms |
| **FPS** | **23.17** |
| Throughput | 23.15 req/s |

## Performance Summary

### TensorRT FP16 vs PyTorch Comparison (640x640, 100 requests)

| Mode | TensorRT FPS | PyTorch FPS | Speedup |
|------|--------------|-------------|---------|
| **LOCAL Stream** ⚡ | **71.43** | 22.42 | **3.2x** |
| **LOCAL Unary** | **77.30** | 20.89 | **3.7x** |
| **API Stream** | **26.90** | 13.20 | **2.0x** |
| **API Unary** | **23.17** | 10.60 | **2.2x** |

### Raw TensorRT Inference (benchmark_tensorrt.py)

| Metric | PyTorch FP32 | TensorRT FP16 |
|--------|--------------|---------------|
| Mean Latency | 18.29ms | 2.16ms |
| **FPS** | 54.7 | **462.7** |
| **Speedup** | 1x | **8.5x** |

*Note: Raw inference measures only model forward pass, excluding pre/post-processing.*

### Key Insights

1. **TensorRT provides 3-4x speedup** for end-to-end local inference
2. **API speedup is ~2x** (network overhead dominates)
3. **Raw inference is 8.5x faster** with TensorRT FP16 (462 vs 55 FPS)
4. **First-load penalty**: ~5 minutes to build TensorRT engine from ONNX (one-time)

### Recommendations by Use Case

| Use Case | Recommended Mode | FPS | Latency |
|----------|------------------|-----|---------|
| **Real-time streaming** | LOCAL Stream + TensorRT | 71.43 | 14ms |
| **Maximum throughput** | LOCAL Unary + TensorRT | 77.30 | 13ms |
| **API real-time** | API Stream + TensorRT | 26.90 | 37ms |
| **API standard** | API Unary + TensorRT | 23.17 | 43ms |

## TensorRT Setup

The model automatically builds TensorRT engine from ONNX on first load:

1. **First request**: Builds engine (~5 min), then serves at full speed
2. **Subsequent requests**: Uses cached engine immediately
3. **Fallback**: If TensorRT unavailable, uses PyTorch backend

### Files
- `dfine.onnx` - ONNX model (included with model, 42MB)
- `dfine.engine` - TensorRT engine (built at runtime, cached)

### Benchmark Scripts
- `benchmark_tensorrt.py` - Compare PyTorch vs TensorRT raw inference
- `benchmark_direct.py` - Local unary-unary benchmark
- `benchmark_local_stream.py` - Local stream-stream benchmark
- `benchmark_api_unary.py` - API unary-unary benchmark
- `benchmark_api_stream.py` - API stream-stream benchmark

## Notes
- All benchmarks run with 100 images for statistical reliability
- TensorRT engine is GPU-specific (A10 engine won't work on T4)
- NMS disabled by default (D-FINE is DETR-based, doesn't need it)
- API performance includes network overhead
