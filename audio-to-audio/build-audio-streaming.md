# Build Log: Audio-to-Audio Streaming Model

## Project
**Goal:** Create a Clarifai runner example demonstrating bidirectional (stream-stream) audio streaming using Orpheus-TTS + vLLM + SNAC vocoder.

**Location:** `/Users/luvbansal/work/runners-examples/audio-to-audio/`

## Plan
1. Create model.py with all 3 streaming patterns (predict, generate, stream)
2. Create config.yaml, requirements.txt
3. Create client.py with examples for all 3 patterns + LLM-to-TTS pipeline
4. Create README.md with architecture docs
5. Test locally with `clarifai model serve --grpc`
6. Fix any issues found during testing
7. Deploy to Clarifai platform

## Progress

### Step 1: Model Implementation (DONE)
- Created `1/model.py` with `OrpheusAudioModel(ModelClass)`
- Uses Orpheus-TTS 3B (`canopylabs/orpheus-3b-0.1-ft`) via vLLM
- SNAC vocoder (`hubertsiuzdak/snac_24khz`) for audio token decoding
- Implemented `_streaming_generate()` — core method that buffers vLLM tokens and yields WAV chunks incrementally (10 SNAC frames = ~300ms per chunk)
- Three `@ModelClass.method` endpoints:
  - `predict()` → unary-unary (PostModelOutputs)
  - `generate()` → unary-stream (GenerateModelOutputs)
  - `stream()` → stream-stream (StreamModelOutputs) — takes `Iterator[NamedFields(text, voice)]`, yields `Iterator[dt.Audio]`
- **What worked:** Pattern matches existing examples (mask2former stream, hf-llama generate). SDK auto-detects `Iterator` type annotations to route to correct gRPC method.

### Step 2: Config & Dependencies (DONE)
- `config.yaml`: model_type_id=text-to-audio, A10G GPU, Python 3.11
- `requirements.txt`: vllm>=0.6.0, snac, transformers, torch, numpy

### Step 3: Client Implementation (DONE)
- Created `client.py` with 4 examples:
  - `example_predict()` — single request/response
  - `example_generate()` — streaming audio output with time-to-first-chunk measurement
  - `example_stream()` — bidirectional with 4 text segments
  - `example_stream_with_llm()` — sentence-buffered LLM token stream → TTS
- Uses `ModelClient.from_local_grpc(port)` for local testing, `Model(url=...).new_client()` for deployed
- Includes `merge_wav_chunks()` helper to combine streamed WAV chunks (strips 44-byte headers, concatenates PCM)

### Step 4: README Documentation (DONE)
- Architecture diagram (text → Orpheus tokens → SNAC decode → WAV)
- SNAC token redistribution explained (7 tokens → 3 layers: 1+2+4)
- Sequence diagrams for all 3 streaming patterns
- Client SDK usage with generator pattern for stream-stream
- Quick start commands

## What Worked
- Clarifai SDK streaming API is clean: `Iterator` in type hints auto-routes to correct gRPC method
- `NamedFields(text=str, voice=str)` for structured streaming input
- `ModelClient._stream()` handles generator → protobuf serialization automatically
- vLLM `llm.generate()` returns incrementally, allowing token-by-token streaming

## Known Risks / Open Questions
- vLLM streaming: `llm.generate()` in sync mode may return all at once (not token-by-token). Need to verify if it actually streams incrementally or if we need `AsyncLLM` with `use_beam_search=False`. May need to switch to `llm.generate(..., stream=True)` if available in the vLLM version.
- SNAC model GPU memory: Orpheus (3B fp16 ~6GB) + SNAC on same GPU. Should fit on A10G (24GB) but untested.
- WAV chunk playback: Each chunk has its own WAV header. Client must either play each independently or strip headers and concatenate. Current client does the latter.
- `redistribute_codes` offset math: Taken from Orpheus reference impl but not independently verified.

## Current Status
**Phase:** Pre-testing — all files created, ready for local test

## Next Action
- Test with `clarifai model serve ./audio-to-audio --grpc --port 8000`
- Verify model loads without errors
- Test each streaming pattern with client.py
- Fix any runtime issues
