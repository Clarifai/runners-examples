"""
Client examples for Audio-to-Audio Streaming Model.

Demonstrates all three Clarifai streaming patterns from the client side:
  1. predict()   - Unary-Unary:       single text -> single audio
  2. generate()  - Unary-Stream:      single text -> streaming audio chunks
  3. stream()    - Stream-Stream:      streaming text -> streaming audio (bidirectional)

Usage:
  # Against deployed model on Clarifai platform
  python client.py --mode predict --text "Hello world"
  python client.py --mode generate --text "Hello world"
  python client.py --mode stream

  # Against local gRPC server (clarifai model serve --grpc)
  python client.py --mode predict --text "Hello world" --local --port 8000
"""

import argparse
import os
import time
import wave

from clarifai.client import Model
from clarifai.client.model_client import ModelClient


def save_audio(audio_bytes: bytes, filepath: str):
  """Save raw WAV bytes to a file."""
  with open(filepath, "wb") as f:
    f.write(audio_bytes)
  print(f"  Saved: {filepath} ({len(audio_bytes)} bytes)")


def merge_wav_chunks(chunks: list[bytes], output_path: str):
  """Merge multiple WAV byte chunks into a single WAV file."""
  all_pcm = b""
  sample_rate = 24000
  for chunk in chunks:
    if len(chunk) > 44:
      all_pcm += chunk[44:]  # Skip WAV header (44 bytes)

  # Write merged WAV
  with wave.open(output_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sample_rate)
    wf.writeframes(all_pcm)
  print(f"  Merged: {output_path} ({len(all_pcm)} PCM bytes)")


# ─── Example 1: predict() — Unary-Unary ─────────────────────────────────────
def example_predict(client: ModelClient, text: str, voice: str):
  """
  Single request, single response.

  Sends text, waits for full audio generation, receives complete WAV.
  Simplest pattern — good for short utterances where latency isn't critical.
  """
  print(f"\n{'='*60}")
  print("Example 1: predict() — Unary-Unary")
  print(f"{'='*60}")
  print(f"  Text:  '{text}'")
  print(f"  Voice: {voice}")

  start = time.time()
  result = client.predict(text=text, voice=voice)
  elapsed = time.time() - start

  # result is an Audio object with .bytes
  audio_bytes = result.bytes
  print(f"  Time:  {elapsed:.2f}s")
  print(f"  Audio: {len(audio_bytes)} bytes")

  save_audio(audio_bytes, "output_predict.wav")


# ─── Example 2: generate() — Unary-Stream ───────────────────────────────────
def example_generate(client: ModelClient, text: str, voice: str):
  """
  Single request, streaming response.

  Sends text once, receives audio chunks as they're generated.
  Low time-to-first-audio — client can start playback immediately.
  """
  print(f"\n{'='*60}")
  print("Example 2: generate() — Unary-Stream")
  print(f"{'='*60}")
  print(f"  Text:  '{text}'")
  print(f"  Voice: {voice}")

  chunks = []
  start = time.time()
  first_chunk_time = None

  for i, result in enumerate(client.generate(text=text, voice=voice)):
    if first_chunk_time is None:
      first_chunk_time = time.time() - start
      print(f"  Time to first chunk: {first_chunk_time:.2f}s")

    audio_bytes = result.bytes
    chunks.append(audio_bytes)
    print(f"  Chunk {i}: {len(audio_bytes)} bytes")

  elapsed = time.time() - start
  print(f"  Total time: {elapsed:.2f}s")
  print(f"  Total chunks: {len(chunks)}")

  merge_wav_chunks(chunks, "output_generate.wav")


# ─── Example 3: stream() — Stream-Stream (Bidirectional) ────────────────────
def example_stream(client: ModelClient, voice: str):
  """
  Bidirectional streaming — the main attraction.

  Client sends a stream of text segments (simulating sentences arriving from
  an LLM or real-time transcription). Server streams back audio for each
  segment as it's synthesized. Both directions are active simultaneously.

  This is what enables real-time voice agents:
    LLM generates text -> stream to TTS -> stream audio to user
  """
  print(f"\n{'='*60}")
  print("Example 3: stream() — Stream-Stream (Bidirectional)")
  print(f"{'='*60}")

  # Simulate text arriving in chunks (e.g., from an LLM streaming response)
  text_segments = [
      "Welcome to the audio streaming demo.",
      "This sentence was generated while you were still hearing the first one.",
      "That's the power of bidirectional streaming.",
      "Each text chunk is synthesized independently and streamed back.",
  ]

  def input_generator():
    """Simulate a stream of text inputs arriving over time."""
    for i, text in enumerate(text_segments):
      print(f"  >> Sending chunk {i}: '{text}'")
      yield {"text": text, "voice": voice}

  chunks = []
  start = time.time()
  first_chunk_time = None

  # client.stream() takes the streaming arg as a generator
  for i, result in enumerate(client.stream(input_stream=input_generator())):
    if first_chunk_time is None:
      first_chunk_time = time.time() - start
      print(f"  Time to first audio: {first_chunk_time:.2f}s")

    audio_bytes = result.bytes
    chunks.append(audio_bytes)
    print(f"  << Audio chunk {i}: {len(audio_bytes)} bytes")

  elapsed = time.time() - start
  print(f"  Total time: {elapsed:.2f}s")
  print(f"  Total audio chunks: {len(chunks)}")

  merge_wav_chunks(chunks, "output_stream.wav")


# ─── Example 4: stream() with live LLM integration ──────────────────────────
def example_stream_with_llm(client: ModelClient, voice: str):
  """
  Real-world pattern: pipe an LLM's streaming text output into TTS streaming.

  This shows the killer use case — an LLM generates text token-by-token,
  we buffer into sentences, and stream each sentence to TTS as it completes.
  The user hears audio with minimal delay after the LLM starts responding.
  """
  print(f"\n{'='*60}")
  print("Example 4: stream() with simulated LLM output")
  print(f"{'='*60}")

  # Simulated LLM streaming output (in production, this would come from
  # client.generate() on an LLM model)
  llm_tokens = [
      "The ", "future ", "of ", "AI ", "is ", "incredibly ", "exciting. ",
      "We're ", "seeing ", "breakthroughs ", "in ", "reasoning, ",
      "multimodal ", "understanding, ", "and ", "real-time ", "interaction. ",
      "Voice ", "interfaces ", "will ", "become ", "the ", "primary ",
      "way ", "we ", "communicate ", "with ", "AI ", "systems. ",
  ]

  def sentence_buffered_generator():
    """Buffer LLM tokens into sentences, yield each as a stream chunk."""
    buffer = ""
    sentence_endings = {".", "!", "?"}

    for token in llm_tokens:
      buffer += token
      # Yield when we hit a sentence boundary
      if buffer.rstrip() and buffer.rstrip()[-1] in sentence_endings:
        sentence = buffer.strip()
        print(f"  >> LLM sentence: '{sentence}'")
        yield {"text": sentence, "voice": voice}
        buffer = ""

    # Flush any remaining text
    if buffer.strip():
      print(f"  >> LLM remainder: '{buffer.strip()}'")
      yield {"text": buffer.strip(), "voice": voice}

  chunks = []
  start = time.time()

  for i, result in enumerate(client.stream(input_stream=sentence_buffered_generator())):
    audio_bytes = result.bytes
    chunks.append(audio_bytes)
    elapsed_chunk = time.time() - start
    print(f"  << Audio chunk {i} at {elapsed_chunk:.2f}s: {len(audio_bytes)} bytes")

  elapsed = time.time() - start
  print(f"  Total time: {elapsed:.2f}s")

  merge_wav_chunks(chunks, "output_stream_llm.wav")


def get_client(args) -> ModelClient:
  """Create a ModelClient for the audio model."""
  if args.local:
    # Connect to local gRPC server
    print(f"Connecting to local gRPC server on port {args.port}...")
    return ModelClient.from_local_grpc(port=args.port)
  else:
    # Connect to deployed model on Clarifai platform
    model_url = args.model_url
    if not model_url:
      raise ValueError(
          "Provide --model-url for deployed model, "
          "or use --local for local testing"
      )
    print(f"Connecting to model: {model_url}")
    model = Model(url=model_url)
    return model.new_client()


def main():
  parser = argparse.ArgumentParser(description="Audio-to-Audio Streaming Client")
  parser.add_argument(
      "--mode",
      choices=["predict", "generate", "stream", "stream-llm", "all"],
      default="all",
      help="Which streaming pattern to demonstrate",
  )
  parser.add_argument("--text", default="Hello! This is a streaming audio demo.", help="Text to synthesize")
  parser.add_argument("--voice", default="tara", help="Speaker voice")
  parser.add_argument("--model-url", help="Clarifai model URL (e.g., https://clarifai.com/user/app/models/model-id)")
  parser.add_argument("--local", action="store_true", help="Connect to local gRPC server")
  parser.add_argument("--port", type=int, default=8000, help="Local gRPC server port")

  args = parser.parse_args()
  client = get_client(args)

  if args.mode in ("predict", "all"):
    example_predict(client, args.text, args.voice)

  if args.mode in ("generate", "all"):
    example_generate(client, args.text, args.voice)

  if args.mode in ("stream", "all"):
    example_stream(client, args.voice)

  if args.mode in ("stream-llm", "all"):
    example_stream_with_llm(client, args.voice)

  print(f"\nDone! Check output_*.wav files.")


if __name__ == "__main__":
  main()
