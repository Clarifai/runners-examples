# D-Fine Object Detection Benchmark Results

## Test Configuration
- **Image Size**: 640x640 (native D-FINE resolution)
- **Number of Requests**: 100 per test
- **Model**: ustc-community/dfine-small-obj2coco
- **Device**: CUDA (GPU - NVIDIA A10)
- **Backend**: TensorRT (with PyTorch fallback)

## TensorRT FP32 Results (Current)

### 1. LOCAL Stream-Stream
**Command**: `python benchmark_local_stream.py`

| Metric | Value |
|--------|-------|
| Avg Inter-arrival | 10.97ms |
| Min Inter-arrival | 9.79ms |
| Max Inter-arrival | 18.13ms |
| Std Dev | 1.33ms |
| **FPS** | **91.13** |
| Overall Throughput | 90.93 images/s |
| Time to First Result | 13.45ms |
| Time to Last Result | 1099.75ms |

### 2. LOCAL Unary-Unary (Direct Model Calls)
**Command**: `python benchmark_direct.py`

| Metric | Value |
|--------|-------|
| Average Latency | 11.86ms |
| Min Latency | 10.25ms |
| Max Latency | 49.15ms |
| Std Dev | 4.26ms |
| **FPS** | **84.35** |
| Throughput | 84.30 req/s |

---

## TensorRT FP16 Results (Previous)

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

### TensorRT FP32 vs FP16 Comparison (640x640, 100 requests)

| Mode | TensorRT FP32 | TensorRT FP16 | Difference |
|------|---------------|---------------|------------|
| **LOCAL Stream** ⚡ | **91.13 FPS** | 71.43 FPS | **+28%** |
| **LOCAL Unary** | **84.35 FPS** | 77.30 FPS | **+9%** |

*Note: FP32 is faster than FP16 on this GPU (A10). FP16 also had numerical stability issues causing NaN outputs on some inputs.*

### TensorRT FP32 vs PyTorch Comparison (640x640, 100 requests)

| Mode | TensorRT FP32 FPS | PyTorch FPS | Speedup |
|------|-------------------|-------------|---------|
| **LOCAL Stream** ⚡ | **91.13** | 22.42 | **4.1x** |
| **LOCAL Unary** | **84.35** | 20.89 | **4.0x** |
| **API Stream** | TBD | 13.20 | - |
| **API Unary** | TBD | 10.60 | - |

### TensorRT FP16 vs PyTorch Comparison (640x640, 100 requests)

| Mode | TensorRT FP16 FPS | PyTorch FPS | Speedup |
|------|-------------------|-------------|---------|
| **LOCAL Stream** | 71.43 | 22.42 | **3.2x** |
| **LOCAL Unary** | 77.30 | 20.89 | **3.7x** |
| **API Stream** | 26.90 | 13.20 | **2.0x** |
| **API Unary** | 23.17 | 10.60 | **2.2x** |

### Key Insights

1. **TensorRT FP32 outperforms FP16** on NVIDIA A10 for this model
2. **FP32 provides ~4x speedup** over PyTorch for end-to-end local inference
3. **FP16 had stability issues** - NaN outputs on some inputs, so FP32 is recommended
4. **First-load penalty**: ~2 minutes to build TensorRT engine from ONNX (one-time)

### Recommendations by Use Case

| Use Case | Recommended Mode | FPS | Latency |
|----------|------------------|-----|---------|
| **Real-time streaming** | LOCAL Stream + TensorRT FP32 | 91.13 | 11ms |
| **Maximum throughput** | LOCAL Unary + TensorRT FP32 | 84.35 | 12ms |
| **API real-time** | API Stream + TensorRT | TBD | TBD |
| **API standard** | API Unary + TensorRT | TBD | TBD |

## TensorRT Setup

The model automatically builds TensorRT FP32 engine from ONNX on first load:

1. **First request**: Builds engine (~2 min), then serves at full speed
2. **Subsequent requests**: Uses cached engine immediately
3. **Fallback**: If TensorRT unavailable, uses PyTorch backend

### Files
- `dfine.onnx` - ONNX model (included with model, 42MB)
- `dfine_fp32.engine` - TensorRT FP32 engine (built at runtime, cached)

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
