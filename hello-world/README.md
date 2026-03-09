# Hello World Model

A minimal starter example demonstrating how to create and deploy a custom model on Clarifai. This model performs simple string manipulation operations.

## Overview

This example shows how to:
- Inherit from Clarifai's `ModelClass`
- Implement model methods: `predict`, `generate`, and `stream`
- Test locally and deploy to Clarifai

## Prerequisites

- Python 3.11+
- A Clarifai account

## Quickstart

### 1. Install the Clarifai Python SDK

```bash
pip install clarifai
```

### 2. Login to Clarifai

```bash
clarifai login
```

### 3. Test Locally

```bash
clarifai model serve hello-world
```

This starts a local gRPC server so you can test your model before deploying.

### 4. Deploy to Clarifai

```bash
clarifai model deploy hello-world
```

This builds, uploads, and deploys your model to the Clarifai platform in one step.

## Inference

Once deployed, you can run predictions using the Clarifai SDK:

```python
from clarifai.client import Model

model = Model("https://clarifai.com/{user_id}/{app_id}/models/hello-world")

response = model.predict(prompt="What is the future of AI?")
print(response)
```

## Model Structure

```
hello-world/
├── 1/
│   └── model.py         # Model implementation
├── config.yaml          # Deployment config
└── requirements.txt     # Dependencies
```

## Custom Model Development

To create your own model:
1. Modify the model class in `1/model.py`
2. Implement your custom logic in the model methods
3. Update `config.yaml` with your model ID and compute instance
4. Test locally with `clarifai model serve`
5. Deploy with `clarifai model deploy`

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Model Not Loading**: Check that your model class inherits from `ModelClass` and implements `load_model()`

### Getting Help
- [Clarifai Documentation](https://docs.clarifai.com/)
- `clarifai --help` for CLI command help
- `clarifai model --help` for model-specific commands
