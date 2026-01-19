# D-Fine Object Detection Benchmark Results

## Test Configuration
- **Image Size**: 1024x1024
- **Number of Requests**: 100 per test
- **Model**: ustc-community/dfine-small-obj2coco
- **Device**: CUDA (GPU - NVIDIA A10)

## Local Benchmarks

### 1. LOCAL Stream-Stream (Optimized)
**Command**: `python benchmark_local_stream.py`

| Metric | Value |
|--------|-------|
| Avg Inter-arrival | 44.60ms |
| Min Inter-arrival | 32.43ms |
| Max Inter-arrival | 92.47ms |
| Std Dev | 16.08ms |
| **FPS** | **22.42** |
| Overall Throughput | 22.46 images/s |
| Time to First Result | 36.78ms |
| Time to Last Result | 4452.56ms |

**Note**: Uses `batch_size=1` by default for real-time streaming (zero batching latency).

### 2. LOCAL Unary-Unary (Direct Model Calls)
**Command**: `python benchmark_direct.py`

| Metric | Value |
|--------|-------|
| Average Latency | 47.87ms |
| Min Latency | 33.06ms |
| Max Latency | 99.50ms |
| Std Dev | 17.22ms |
| **FPS** | **20.89** |
| Throughput | 20.88 req/s |

## API Benchmarks

### 3. API Stream-Stream
**Command**: `export CLARIFAI_PAT='your_pat' && python benchmark_api_stream.py`

| Metric | Value |
|--------|-------|
| Avg Inter-arrival | 75.74ms |
| Min Inter-arrival | 62.06ms |
| Max Inter-arrival | 214.45ms |
| Std Dev | 16.54ms |
| **FPS** | **13.20** |
| Overall Throughput | 13.19 images/s |
| Time to First Result | 83.28ms |
| Time to Last Result | 7581.36ms |

**Note**: Falls back to sequential API calls (true streaming API not available in SDK).

### 4. API Unary-Unary
**Command**: `export CLARIFAI_PAT='your_pat' && python benchmark_api_unary.py`

| Metric | Value |
|--------|-------|
| Average Latency | 94.38ms |
| Min Latency | 62.88ms |
| Max Latency | 295.36ms |
| Std Dev | 42.39ms |
| **FPS** | **10.60** |
| Throughput | 10.59 req/s |

## Summary

### Performance Comparison (1024x1024 images, 100 requests)

| Mode | FPS | Avg Latency | Speedup vs API Unary |
|------|-----|-------------|----------------------|
| **LOCAL Stream** ⚡ | **22.42** | 44.60ms | **2.12x** |
| LOCAL Unary | 20.89 | 47.87ms | 1.97x |
| API Stream | 13.20 | 75.74ms | 1.25x |
| API Unary | 10.60 | 94.38ms | 1.00x (baseline) |

### Key Insights

1. **Local Stream is fastest**: 22.42 FPS with optimized batching (batch_size=1 for real-time)
2. **Local vs API**: Local is ~2x faster than API (22.42 vs 10.60 FPS)
3. **Stream advantage**: Stream mode is faster than unary for both local and API
   - Local: Stream 7% faster than unary (22.42 vs 20.89 FPS)
   - API: Stream 25% faster than unary (13.20 vs 10.60 FPS)
4. **Real-time ready**: 44.6ms latency with batch_size=1 (default) is excellent for streaming video

### Recommendations by Use Case

| Use Case | Recommended Mode | FPS | Reason |
|----------|------------------|-----|--------|
| **Real-time streaming** | LOCAL Stream (batch=1) | 22.42 | Lowest latency, highest FPS |
| **Offline batch processing** | LOCAL Stream (batch=4) | ~35.46 | 1.83x faster with batching |
| **API real-time** | API Stream | 13.20 | Better than API unary |
| **API batch** | API Stream | 13.20 | Best API performance |

### Batching Analysis

Stream mode supports configurable `batch_size` parameter:
- `batch_size=1` (default): 22.42 FPS, 0ms batching wait - **Use for real-time streaming**
- `batch_size=2`: 27.51 FPS, ~32ms first-frame wait
- `batch_size=4`: 35.46 FPS, ~73ms first-frame wait - **Use for offline processing**
- `batch_size=8`: 31.21 FPS, ~186ms first-frame wait

**For real-time video**: Always use `batch_size=1` to avoid latency penalties.

## Files
- `benchmark_direct.py` - Local unary-unary benchmark
- `benchmark_local_stream.py` - Local stream-stream benchmark (supports batching)
- `benchmark_api_unary.py` - API unary-unary benchmark
- `benchmark_api_stream.py` - API stream-stream benchmark
- `benchmark_batched_stream.py` - Tests different batch sizes
- `benchmark_streaming_latency.py` - Analyzes latency vs throughput tradeoff
- `run_model.sh` - Helper script to start the local runner (with Ctrl+C support)
- `stop_model.sh` - Helper script to stop all running model processes

## Notes
- All benchmarks run with 100 images for statistical reliability
- Local benchmarks use the same model instance to avoid reloading overhead
- API benchmarks require a valid CLARIFAI_PAT environment variable and running local runner
- API performance includes network overhead
- Stream implementation now optimized: was 10.98 FPS, now 22.42 FPS (2x improvement)
