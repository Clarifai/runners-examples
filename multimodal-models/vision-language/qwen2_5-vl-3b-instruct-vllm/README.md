---
complexity: intermediate
framework: vllm
model_size: 3B
gpu_required: true
min_gpu_memory: 20Gi
features: [vision-language, chat, streaming]
model_class: OpenAIModelClass
task: vision-language
---

# Qwen2_5 Vl 3B Instruct Vllm

Example demonstrating vision-language using vllm.

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

- vision-language
- chat
- streaming
