# Audio-to-Audio Streaming with Orpheus-TTS

A complete example of **bidirectional audio streaming** on the Clarifai platform using [Orpheus-TTS](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) (3B parameter text-to-speech model) served via vLLM with SNAC vocoder decoding.

This example demonstrates all three Clarifai API streaming patterns — unary-unary, unary-stream, and **stream-stream (bidirectional)** — for real-time voice applications.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Model Pipeline                           │
│                                                                 │
│   Text ──> Orpheus Tokenizer ──> vLLM (Orpheus-3B) ──> Audio   │
│            (prompt tokens)       (audio tokens)       Tokens    │
│                                                         │       │
│                                          ┌──────────────┘       │
│                                          ▼                      │
│                                   SNAC Vocoder                  │
│                                   (24kHz decoder)               │
│                                          │                      │
│                                          ▼                      │
│                                    WAV Audio                    │
│                                    (streamed)                   │
└─────────────────────────────────────────────────────────────────┘
```

**Two models work together:**
1. **Orpheus-TTS** (via vLLM) — A causal language model fine-tuned to generate audio token sequences from text. It outputs special tokens (IDs >= 128266) that encode audio.
2. **SNAC** (24kHz vocoder) — Decodes audio tokens into waveform audio. Uses a 3-layer hierarchical structure where every 7 tokens produce ~30ms of audio.

## How the Model Works (`model.py`)

### Token Constants

```python
START_TOKEN = 128259        # Marks beginning of Orpheus prompt
END_TOKENS = {128009, 128260}  # Signals end of generation
AUDIO_TOKEN_OFFSET = 128266    # Tokens >= this are audio codes
SNAC_TOKENS_PER_FRAME = 7     # 7 tokens = 1 SNAC frame (~30ms audio)
```

Orpheus extends a standard LLM vocabulary with audio tokens. Tokens below 128266 are text/control tokens; tokens at or above 128266 are audio codes. We subtract the offset to get raw SNAC codes (0-4095 per layer).

### SNAC Token Redistribution (`redistribute_codes`)

Orpheus outputs audio tokens in a flat sequence, but SNAC expects them split across 3 hierarchical layers. Every 7 consecutive tokens map to:

```
Token index:  [0]    [1]     [2]      [3]      [4]     [5]      [6]
Layer:        L1     L2      L3       L3       L2      L3       L3
Offset:       0    -4096   -8192   -12288   -16384  -20480   -24576
```

- **Layer 1** (coarse): 1 token per frame — captures overall spectral shape
- **Layer 2** (mid): 2 tokens per frame — adds harmonic detail
- **Layer 3** (fine): 4 tokens per frame — adds high-frequency texture

The SNAC decoder takes these 3 layers and synthesizes a 24kHz audio waveform.

### WAV Encoding (`audio_array_to_wav_bytes`)

Converts the float32 numpy array from SNAC into self-contained WAV bytes with a standard 44-byte header (RIFF/WAVE format, mono, 16-bit PCM, 24kHz). Each streamed chunk is a valid WAV file that can be played independently.

### Model Initialization (`load_model`)

```python
def load_model(self):
    # 1. Load Orpheus-TTS via vLLM for fast batched token generation
    self.llm = LLM(model="canopylabs/orpheus-3b-0.1-ft", ...)

    # 2. Load SNAC vocoder on GPU for audio decoding
    self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")
```

vLLM provides optimized inference with PagedAttention, continuous batching, and streaming output — critical for low-latency audio generation.

### Prompt Building (`_build_prompt_tokens`)

Orpheus uses a specific prompt format:

```
<custom_token_3>{voice}: {text}<custom_token_4>
```

The custom tokens are replaced with Orpheus-specific IDs:
- `<custom_token_3>` -> `128259` (START_TOKEN)
- `<custom_token_4>` -> `128009` (end marker)

Available voices: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`.

### Streaming Audio Generation (`_streaming_generate`)

This is the core method that enables low-latency streaming. Instead of generating all tokens first and then decoding, it processes tokens **incrementally**:

```
vLLM generates tokens one-by-one
         │
         ▼
   ┌─────────────┐
   │ Token Buffer │  Accumulate audio tokens
   └─────┬───────┘
         │ When buffer has 70 tokens (10 frames x 7 tokens/frame)
         ▼
   ┌─────────────┐
   │ SNAC Decode  │  Decode to ~300ms of audio
   └─────┬───────┘
         │
         ▼
   yield WAV chunk   Client receives and plays immediately
         │
         ▼
   Continue buffering next tokens...
```

**Key parameters:**
- `frames_per_chunk=10`: Each yield contains 10 SNAC frames = 70 tokens = ~300ms audio. Lower = less latency but more overhead; higher = smoother playback but more delay.
- The method tracks `prev_num_tokens` to extract only newly generated tokens each iteration.
- End tokens (128009, 128260) signal generation is complete.
- Any remaining tokens (< 70 but >= 7) are flushed as a final chunk.

## Three Streaming Patterns

### 1. `predict()` — Unary-Unary

```
Client                          Server
  │                               │
  │── POST text ─────────────────>│
  │                               │  Generate all audio
  │                               │  Merge WAV chunks
  │<──────────── complete WAV ────│
```

**gRPC RPC:** `PostModelOutputs`

Simplest pattern. Sends text, waits for full generation, receives one complete WAV file. Good for short utterances where latency isn't critical.

The implementation collects all streaming chunks internally, strips WAV headers (44 bytes each), concatenates raw PCM data, and re-encodes as a single WAV.

### 2. `generate()` — Unary-Stream

```
Client                          Server
  │                               │
  │── POST text ─────────────────>│
  │                               │  Generate tokens...
  │<──── WAV chunk 0 (~300ms) ────│  ← first audio arrives quickly
  │<──── WAV chunk 1 (~300ms) ────│
  │<──── WAV chunk 2 (~300ms) ────│
  │         ...                   │
  │<──── WAV chunk N (remainder) ─│
```

**gRPC RPC:** `GenerateModelOutputs`

Single request, streaming response. The client sends text once and receives audio chunks as they're generated. Low time-to-first-audio — playback can start before the full response is ready.

Directly yields from `_streaming_generate()`, wrapping each WAV bytes chunk as `dt.Audio`.

### 3. `stream()` — Stream-Stream (Bidirectional)

```
Client                          Server
  │                               │
  │── {text: "Hello", voice} ────>│
  │<──── WAV chunk 0 ─────────────│  ← audio for "Hello"
  │<──── WAV chunk 1 ─────────────│
  │── {text: "World", voice} ────>│  ← client sends next while receiving
  │<──── WAV chunk 2 ─────────────│  ← audio for "World"
  │<──── WAV chunk 3 ─────────────│
  │── (close stream) ────────────>│
  │<──── (stream done) ───────────│
```

**gRPC RPC:** `StreamModelOutputs`

Both input and output stream simultaneously over a single gRPC connection. The client sends a stream of `NamedFields(text=str, voice=str)` — each representing a text segment (e.g., a sentence from an LLM). The server synthesizes each segment and streams back audio chunks as they're ready.

Input uses `Iterator[NamedFields(...)]` — the Clarifai framework detects the `Iterator` type annotation and routes it to the `StreamModelOutputs` gRPC method automatically.

**Real-world use cases:**
- **Conversational AI:** LLM generates text token-by-token -> buffer into sentences -> stream to TTS -> user hears audio with minimal delay
- **Live captioning playback:** Real-time transcription text -> streaming audio
- **Interactive voice agents:** Turn-by-turn audio responses over a persistent connection

## Client Examples (`client.py`)

The client demonstrates all patterns using the Clarifai SDK's `ModelClient`:

| Example | Method | Pattern | Output File |
|---------|--------|---------|-------------|
| `example_predict()` | `client.predict(text=..., voice=...)` | Unary-Unary | `output_predict.wav` |
| `example_generate()` | `for chunk in client.generate(text=..., voice=...)` | Unary-Stream | `output_generate.wav` |
| `example_stream()` | `for chunk in client.stream(input_stream=generator())` | Stream-Stream | `output_stream.wav` |
| `example_stream_with_llm()` | `client.stream()` with sentence-buffered LLM output | Stream-Stream | `output_stream_llm.wav` |

### How the client-side stream works

For bidirectional streaming, the SDK expects a **Python generator** for the streaming input parameter:

```python
def input_generator():
    yield {"text": "First sentence.", "voice": "tara"}
    yield {"text": "Second sentence.", "voice": "tara"}

for audio_chunk in client.stream(input_stream=input_generator()):
    play(audio_chunk.bytes)  # WAV audio
```

Under the hood, the SDK:
1. Identifies `input_stream` as the streaming parameter (from method signatures)
2. Serializes each yielded dict into a `PostModelOutputsRequest` protobuf
3. Sends them over a gRPC bidirectional stream (`StreamModelOutputs`)
4. Deserializes each response into `dt.Audio` and yields it back

### LLM-to-TTS pipeline pattern

The `example_stream_with_llm()` shows the real-world killer pattern — buffering LLM tokens into sentences and piping them to TTS:

```python
def sentence_buffered_generator():
    buffer = ""
    for token in llm_streaming_output:
        buffer += token
        if buffer.rstrip()[-1] in ".!?":
            yield {"text": buffer.strip(), "voice": "tara"}
            buffer = ""
```

## Quick Start

### Local Testing

```bash
# Start the model server
clarifai model serve ./audio-to-audio --grpc --port 8000

# In another terminal, run the client
python client.py --local --port 8000 --mode all
```

### Deploy to Clarifai

```bash
# Upload the model
clarifai model upload ./audio-to-audio

# Run client against deployed model
python client.py --model-url https://clarifai.com/USER/APP/models/orpheus-tts-streaming --mode stream
```

### Individual patterns

```bash
# Unary-unary
python client.py --local --mode predict --text "Hello world" --voice tara

# Unary-stream (streaming audio output)
python client.py --local --mode generate --text "A longer passage to stream back as audio."

# Stream-stream (bidirectional)
python client.py --local --mode stream

# Stream-stream with simulated LLM integration
python client.py --local --mode stream-llm
```

## File Structure

```
audio-to-audio/
├── 1/
│   └── model.py           # Model: Orpheus-TTS with predict/generate/stream
├── config.yaml            # Clarifai deployment config (A10G GPU)
├── requirements.txt       # Python dependencies
├── client.py              # Client: 4 examples covering all patterns
└── README.md              # This file
```

## Requirements

- **GPU:** NVIDIA A10G or better (model is ~6GB in fp16)
- **Python:** 3.11+
- **Key dependencies:** vLLM, SNAC, transformers, torch
