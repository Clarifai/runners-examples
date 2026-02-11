---
complexity: beginner
framework: vllm
model_size: 3.8B
gpu_required: true
min_gpu_memory: 20Gi
features: [text-generation, chat, streaming]
model_class: OpenAIModelClass
task: text-generation
---

# Vllm Phi 3.5 Mini Instruct

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
- Minimum GPU Memory: 20Gi
- Framework: vllm

## Features

- text-generation
- chat
- streaming
