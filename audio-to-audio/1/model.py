"""
Audio-to-Audio Streaming Model Example

Demonstrates all three Clarifai streaming patterns using Orpheus-TTS + vLLM:
  - predict():   text -> Audio                              (UNARY_UNARY)
  - generate():  text -> Iterator[Audio]                    (UNARY_STREAMING)
  - stream():    Iterator[NamedFields] -> Iterator[Audio]   (STREAMING_STREAMING)

Uses vLLM's streaming output to decode audio tokens incrementally as they're
generated — yielding audio chunks with minimal latency instead of waiting for
full generation to complete.

Reference: https://vllm.ai/blog/streaming-realtime
"""

import io
import struct
from typing import Iterator

import numpy as np
import torch
from snac import SNAC
from vllm import LLM, SamplingParams

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils import data_types as dt
from clarifai.runners.utils.data_types import NamedFields
from clarifai.utils.logging import logger

# Orpheus TTS special tokens
START_TOKEN = 128259
END_TOKENS = {128009, 128260}
AUDIO_TOKEN_OFFSET = 128266
# Number of SNAC tokens per frame (7 tokens = ~30ms audio at 24kHz)
SNAC_TOKENS_PER_FRAME = 7


def redistribute_codes(code_list, snac_model):
  """Convert flat Orpheus token list into SNAC-compatible 3-layer structure."""
  layer_1, layer_2, layer_3 = [], [], []
  for i in range((len(code_list) + 1) // 7):
    base = 7 * i
    if base + 6 < len(code_list):
      layer_1.append(code_list[base])
      layer_2.append(code_list[base + 1] - 4096)
      layer_3.append(code_list[base + 2] - 2 * 4096)
      layer_3.append(code_list[base + 3] - 3 * 4096)
      layer_2.append(code_list[base + 4] - 4 * 4096)
      layer_3.append(code_list[base + 5] - 5 * 4096)
      layer_3.append(code_list[base + 6] - 6 * 4096)

  codes = [
      torch.tensor(layer_1, device=snac_model.device).unsqueeze(0),
      torch.tensor(layer_2, device=snac_model.device).unsqueeze(0),
      torch.tensor(layer_3, device=snac_model.device).unsqueeze(0),
  ]
  audio = snac_model.decode(codes).squeeze(0).cpu().numpy()
  return audio


def audio_array_to_wav_bytes(audio_np, sample_rate=24000):
  """Convert a numpy float32 audio array to WAV bytes."""
  audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
  buf = io.BytesIO()
  num_samples = len(audio_int16)
  data_size = num_samples * 2

  # WAV header
  buf.write(b"RIFF")
  buf.write(struct.pack("<I", 36 + data_size))
  buf.write(b"WAVE")
  buf.write(b"fmt ")
  buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
  buf.write(b"data")
  buf.write(struct.pack("<I", data_size))
  buf.write(audio_int16.tobytes())
  return buf.getvalue()


class OrpheusAudioModel(ModelClass):
  """
  Audio-to-Audio streaming model using Orpheus-TTS + vLLM.

  Demonstrates all three Clarifai API streaming patterns:
    - predict():   Single text in -> single audio out       (unary-unary)
    - generate():  Single text in -> streaming audio out     (unary-stream)
    - stream():    Streaming text in -> streaming audio out  (stream-stream / bidirectional)
  """

  def load_model(self):
    """Load Orpheus TTS model via vLLM and SNAC vocoder."""
    model_id = "canopylabs/orpheus-3b-0.1-ft"

    logger.info(f"Loading Orpheus TTS model: {model_id}")
    self.llm = LLM(
        model=model_id,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        dtype="float16",
    )
    self.tokenizer = self.llm.get_tokenizer()

    logger.info("Loading SNAC vocoder...")
    self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

    self.sample_rate = 24000
    self.sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    logger.info("Model loaded successfully!")

  def _build_prompt_tokens(self, text: str, voice: str = "tara") -> list:
    """Build Orpheus-format prompt token IDs."""
    prompt = f"<custom_token_3>{voice}: {text}<custom_token_4>"
    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    start_id = self.tokenizer.convert_tokens_to_ids("<custom_token_3>")
    end_id = self.tokenizer.convert_tokens_to_ids("<custom_token_4>")
    return [
        START_TOKEN if t == start_id else 128009 if t == end_id else t
        for t in input_ids
    ]

  def _decode_audio_chunk(self, audio_tokens: list) -> bytes:
    """Decode a list of audio tokens to WAV bytes via SNAC."""
    if len(audio_tokens) < SNAC_TOKENS_PER_FRAME:
      return b""
    # Trim to multiple of 7
    usable = len(audio_tokens) - (len(audio_tokens) % SNAC_TOKENS_PER_FRAME)
    audio_tokens = audio_tokens[:usable]
    with torch.no_grad():
      audio_np = redistribute_codes(audio_tokens, self.snac_model)
    return audio_array_to_wav_bytes(audio_np.flatten(), self.sample_rate)

  def _streaming_generate(self, text: str, voice: str = "tara",
                          max_tokens: int = 4096, frames_per_chunk: int = 10):
    """
    Generate audio from text using vLLM streaming output.

    Yields WAV audio chunks incrementally as tokens are generated — the key
    to low-latency streaming. Each chunk contains `frames_per_chunk` SNAC
    frames (~30ms per frame, so 10 frames ≈ 300ms of audio per yield).

    Args:
      text: Text to synthesize.
      voice: Speaker voice.
      max_tokens: Max tokens to generate.
      frames_per_chunk: SNAC frames per audio chunk (controls latency vs overhead).

    Yields:
      bytes: WAV audio chunks.
    """
    input_ids = self._build_prompt_tokens(text, voice)
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=self.sampling_params.temperature,
        top_p=self.sampling_params.top_p,
        repetition_penalty=self.sampling_params.repetition_penalty,
    )

    # Use vLLM streaming: iterate over partial outputs as tokens are generated
    token_buffer = []
    tokens_per_chunk = SNAC_TOKENS_PER_FRAME * frames_per_chunk
    prev_num_tokens = 0

    for request_output in self.llm.generate(
        prompt_token_ids=[input_ids],
        sampling_params=params,
        use_tqdm=False,
    ):
      # Get newly generated tokens since last iteration
      all_token_ids = request_output.outputs[0].token_ids
      new_tokens = all_token_ids[prev_num_tokens:]
      prev_num_tokens = len(all_token_ids)

      # Check for end tokens
      hit_end = False
      for tok in new_tokens:
        if tok in END_TOKENS:
          hit_end = True
          break
        if tok >= AUDIO_TOKEN_OFFSET:
          token_buffer.append(tok - AUDIO_TOKEN_OFFSET)

      # Yield audio when we have enough tokens for a chunk
      while len(token_buffer) >= tokens_per_chunk:
        chunk_tokens = token_buffer[:tokens_per_chunk]
        token_buffer = token_buffer[tokens_per_chunk:]
        wav_bytes = self._decode_audio_chunk(chunk_tokens)
        if wav_bytes:
          yield wav_bytes

      if hit_end:
        break

    # Flush remaining tokens
    if len(token_buffer) >= SNAC_TOKENS_PER_FRAME:
      wav_bytes = self._decode_audio_chunk(token_buffer)
      if wav_bytes:
        yield wav_bytes

  # ── Unary-Unary: Single request, single response ──────────────────
  @ModelClass.method
  def predict(
      self,
      text: str = "",
      voice: str = "tara",
      max_tokens: int = 4096,
  ) -> dt.Audio:
    """
    Convert text to speech in a single request/response.

    Args:
      text: Text to synthesize.
      voice: Speaker voice (tara, leah, jess, leo, dan, mia, zac, zoe).
      max_tokens: Maximum generation length.

    Returns:
      Audio: Complete WAV audio of the synthesized speech.
    """
    logger.info(f"predict(): text='{text[:80]}', voice={voice}")

    # Collect all streaming chunks into one
    all_chunks = list(self._streaming_generate(text, voice, max_tokens))
    if not all_chunks:
      return dt.Audio(bytes=b"")

    # If single chunk, return directly
    if len(all_chunks) == 1:
      return dt.Audio(bytes=all_chunks[0])

    # Merge multiple WAV chunks into one contiguous WAV
    raw_samples = b""
    for wav_bytes in all_chunks:
      # Skip WAV header (44 bytes) to get raw PCM data
      raw_samples += wav_bytes[44:]

    merged_wav = audio_array_to_wav_bytes(
        np.frombuffer(raw_samples, dtype=np.int16).astype(np.float32) / 32767,
        self.sample_rate,
    )
    return dt.Audio(bytes=merged_wav)

  # ── Unary-Stream: Single request, streaming response ───────────────
  @ModelClass.method
  def generate(
      self,
      text: str = "",
      voice: str = "tara",
      max_tokens: int = 4096,
  ) -> Iterator[dt.Audio]:
    """
    Convert text to speech with streaming audio output.

    Uses vLLM's streaming generation to yield audio chunks incrementally
    as tokens are produced — the client can start playback before the full
    response is ready, giving low time-to-first-audio.

    Args:
      text: Text to synthesize.
      voice: Speaker voice.
      max_tokens: Maximum generation length.

    Yields:
      Audio: Incremental WAV audio chunks (~300ms each).
    """
    logger.info(f"generate(): text='{text[:80]}', voice={voice}")
    for wav_bytes in self._streaming_generate(text, voice, max_tokens):
      yield dt.Audio(bytes=wav_bytes)

  # ── Stream-Stream: Bidirectional streaming (2-way) ─────────────────
  @ModelClass.method
  def stream(
      self,
      input_stream: Iterator[
          NamedFields(
              text=str,
              voice=str,
          )
      ],
  ) -> Iterator[dt.Audio]:
    """
    Bidirectional streaming: receive text chunks, stream back audio.

    This is the 2-way streaming (stream-stream) pattern. The client sends
    a stream of text segments (e.g., sentences from an LLM, real-time
    transcription, or conversational turns), and the server streams back
    audio for each segment as it's synthesized.

    Use cases:
      - Real-time conversational AI (LLM text output -> streaming TTS)
      - Live captioning playback (streaming text -> streaming audio)
      - Interactive voice agents (turn-by-turn audio responses)
      - Podcast/audiobook generation from streaming text

    The client can keep sending new text while receiving audio for previous
    segments — true bidirectional streaming over a single gRPC connection.

    Args:
      input_stream: Stream of NamedFields with:
        - text (str): Text segment to synthesize.
        - voice (str): Speaker voice for this segment.

    Yields:
      Audio: Streaming WAV audio chunks for each input text segment.
    """
    logger.info("stream(): bidirectional streaming started")

    for i, input_chunk in enumerate(input_stream):
      text = input_chunk.get("text", "")
      voice = input_chunk.get("voice", "tara")

      if not text.strip():
        continue

      logger.info(f"stream() chunk {i}: '{text[:60]}'")

      # Stream audio for this text segment with low latency
      for wav_bytes in self._streaming_generate(text, voice, max_tokens=2048):
        yield dt.Audio(bytes=wav_bytes)

    logger.info("stream(): client closed input stream, done")
