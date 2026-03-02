---
complexity: intermediate
framework: vllm
model_size: 4B
gpu_required: true
min_gpu_memory: 48Gi
features: [text-generation, chat, streaming]
model_class: OpenAIModelClass
task: text-generation
---

# vLLM Gemma 3 4B Instruct

Google's Gemma 3 4B model deployed with vLLM for high-throughput text generation.

## Quick Start

```bash
# Deploy this model
clarifai model upload
```

## Model Details

- **Model**: google/gemma-3-4b-it
- **Framework**: vLLM
- **Size**: 4B parameters
- **GPU Memory**: 48Gi recommended

## Features

- High-throughput inference with vLLM
- Chat-optimized instruction tuning
- Streaming support
- OpenAI-compatible API

## Configuration

See `config.yaml` for model configuration details. Key settings:

- Checkpoints downloaded at runtime from HuggingFace
- Requires 48Gi GPU memory
- Supports NVIDIA GPUs

## Usage

Once deployed, use the OpenAI-compatible API:

```python
from clarifai.client.model import Model

model = Model(
    user_id="your-user-id",
    app_id="your-app-id",
    model_id="gemma-3-4b-it"
)

response = model.predict(
    inputs=[{"data": {"text": {"raw": "What is machine learning?"}}}]
)

print(response.outputs[0].data.text.raw)
```
