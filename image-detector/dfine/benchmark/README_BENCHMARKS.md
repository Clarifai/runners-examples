# D-Fine Object Detection Benchmarks

## Quick Start

### Running Local Benchmarks (No API Key Required)

```bash
# 1. Local Stream-Stream (BEST: 22.42 FPS)
python benchmark_local_stream.py

# 2. Local Unary-Unary (20.89 FPS)
python benchmark_direct.py
```

### Running API Benchmarks (Requires CLARIFAI_PAT and Running Local Runner)

```bash
# Start the local runner first
./run_model.sh

# In another terminal, set your Clarifai PAT
export CLARIFAI_PAT='your_pat_token_here'

# 3. API Stream-Stream (13.20 FPS)
python benchmark_api_stream.py

# 4. API Unary-Unary (10.60 FPS)
python benchmark_api_unary.py
```

## Current Results (640x640 images, 100 requests)

| Benchmark Type | FPS | Avg Latency | Notes |
|----------------|-----|-------------|-------|
| LOCAL Stream-Stream | **22.42** ⚡ | 44.60ms | **BEST** - Optimized for real-time |
| LOCAL Unary-Unary | **20.89** | 47.87ms | Direct model calls |
| API Stream-Stream | **13.20** | 75.74ms | API streaming, 25% faster than API unary |
| API Unary-Unary | **10.60** | 94.38ms | Sequential API calls |

### Key Findings
- **Local Stream is fastest**: 22.42 FPS (7% faster than local unary)
- **Local is ~2x faster** than API (22.42 vs 10.60 FPS)
- **Stream beats unary** for both local and API
- **Real-time ready**: 44.6ms latency perfect for streaming video

### For Real-Time Streaming
- Use `batch_size=1` (default) for **zero batching latency**
- Stream mode: 22.42 FPS at 44.6ms per frame
- Ideal for live video detection

### For Offline Batch Processing
- Use `batch_size=4` for **1.83x speedup** (35.46 FPS)
- Trades latency for throughput
- Not recommended for real-time due to 73ms first-frame wait

## Model Management Scripts

### Start the Local Runner (with Ctrl+C support)
```bash
./run_model.sh
```
This starts the Clarifai local runner with proper signal handling. Press Ctrl+C to stop.

### Stop All Running Models
```bash
./stop_model.sh
```
Forcefully stops all running Clarifai runner processes.

## Benchmark Details

### Configuration
- **Image Size**: 640x640 (native D-FINE resolution)
- **Number of Requests**: 100 (for statistical reliability)
- **Model**: ustc-community/dfine-small-obj2coco
- **Device**: CUDA (GPU - NVIDIA A10)

### What Each Benchmark Measures

1. **Unary-Unary**: Sequential individual requests. Measures traditional request/response latency.
2. **Stream-Stream**: Streaming mode where images are processed continuously. Optimized for real-time video.

### Stream Batching Options

The stream implementation supports configurable batching:

```python
# Real-time streaming (default)
runner.stream_image(frames, batch_size=1)  # 22.42 FPS, 0ms wait

# Offline batch processing
runner.stream_image(frames, batch_size=4)  # 35.46 FPS, 73ms wait
```

### Modifying Test Parameters

Edit the configuration at the top of each script:
```python
NUM_REQUESTS = 100  # Change number of test images
IMAGE_SIZE = (640, 640)  # Change image dimensions (640x640 is D-FINE native)
```

## Advanced Benchmarks

### Test Different Batch Sizes
```bash
python benchmark_batched_stream.py
```
Tests batch sizes 1, 2, 4, 8 to find optimal throughput.

### Analyze Latency vs Throughput
```bash
python benchmark_streaming_latency.py
```
Shows the latency penalty of batching in real-time scenarios.

## Troubleshooting

### Model Won't Exit with Ctrl+C
Use the stop script:
```bash
./stop_model.sh
```

### API Benchmarks Fail
1. Make sure local runner is running: `ps aux | grep clarifai`
2. Check PAT is set: `echo $CLARIFAI_PAT`
3. Start runner: `./run_model.sh`

### Connection Refused Errors
The local runner isn't serving. Start it with:
```bash
./run_model.sh
```

## Files

- `benchmark_local_stream.py` - Local stream-stream (FASTEST)
- `benchmark_direct.py` - Local unary-unary
- `benchmark_api_stream.py` - API stream-stream
- `benchmark_api_unary.py` - API unary-unary
- `benchmark_batched_stream.py` - Test different batch sizes
- `benchmark_streaming_latency.py` - Latency vs throughput analysis
- `run_model.sh` - Start local runner with Ctrl+C support
- `stop_model.sh` - Force stop all runners
- `BENCHMARK_RESULTS.md` - Detailed results and analysis
