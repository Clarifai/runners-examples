# Local Runners

This folder contains examples for running models locally on your own machine using the Clarifai CLI.

## Prerequisites

- Python 3.10+
- A Clarifai account

## Quickstart

### 1. Install the Clarifai Python SDK

```bash
pip install -U clarifai
```

### 2. Login to Clarifai

```bash
clarifai login
```

### 3. Initialize a Model

Initialize with Ollama:

```bash
clarifai model init --toolkit ollama --model-name gpt-oss:20b
```

See the [Ollama README](ollama-model-upload/) for more information.

Or 

Create a new model from a toolkit:

```bash
clarifai model init --toolkit vllm --model-name google/gemma-3-1b-it
```

### 4. Test Locally

```bash
clarifai model serve <model-path>
```
This connects your local model to the Clarifai platform for testing while keeping the model running locally.

### 5. Deploy to Clarifai

```bash
clarifai model deploy <model-path>
```

## Inference

Once deployed (or running locally via `serve`), test with the Clarifai SDK:

```python
from clarifai.client import Model

model = Model("https://clarifai.com/{user_id}/{app_id}/models/{model_id}")
response = model.predict(prompt="What is the future of AI?")
print(response)
```

## Custom Model Development

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
