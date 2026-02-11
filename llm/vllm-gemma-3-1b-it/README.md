---
complexity: intermediate
framework: vllm
model_size: 1B
gpu_required: true
min_gpu_memory: 8Gi
features: [text-generation, chat, streaming]
model_class: OpenAIModelClass
task: text-generation
---

# Vllm Gemma 3 1B It

Example demonstrating text-generation using vllm.

## Quick Start

```bash
# Deploy this model
clarifai model upload
```

## Configuration

See `config.yaml` for model configuration details.

## Requirements

- GPU: true
- Minimum GPU Memory: 8Gi
- Framework: vllm

## Features

- text-generation
- chat
- streaming
