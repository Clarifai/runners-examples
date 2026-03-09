# Running Ollama Models on Clarifai

Run any [Ollama](https://ollama.com/) model locally and make it accessible from anywhere via the Clarifai platform.

## Prerequisites

- [Ollama](https://ollama.com/download) installed and running
- Python 3.10+
- A Clarifai account

> **Note for Windows users:** After installing Ollama, restart your machine to ensure environment variables take effect.

## Quickstart

### 1. Install Clarifai

```bash
pip install clarifai
```

### 2. Login

```bash
clarifai login
```

### 3. Initialize model from Ollama

```bash
clarifai model init --toolkit ollama --model-name gpt-oss:20b
```

You can use any model from the [Ollama library](https://ollama.com/library):

| Use Case | Model |
|----------|-------|
| General | `gpt-oss:20b` |
| Multimodal | `llama3.2-vision:latest` |
| Tool calling | `llama3-groq-tool-use:latest` |
| Coding agent | `devstral:latest` |

**Options:**

- `--model-name` — Ollama model name (default: `llama3.2`)
- `--port` — Port for the model server (default: `23333`)
- `--context-length` — Context length (default: `8192`)

Example with custom settings:
```bash
clarifai model init --toolkit ollama --model-name gemma3n --port 8008 --context-length 16000
```

### 4. Test locally

```bash
clarifai model serve .
```

This starts the model locally and connects it to the Clarifai platform for inference.

---

## Inference

### OpenAI-Compatible API

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ['CLARIFAI_PAT'],
)

response = client.chat.completions.create(
    model="https://clarifai.com/{user_id}/{app_id}/models/{model_id}",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I check if a Python object is an instance of a class?"},
    ],
    temperature=0.7,
    stream=True,
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Multimodal (OpenAI-Compatible)

```python
import os
import base64
import requests
from openai import OpenAI

def get_image_base64(image_url):
    response = requests.get(image_url)
    return base64.b64encode(response.content).decode('utf-8')

image_base64 = get_image_base64("https://samples.clarifai.com/cat1.jpeg")

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ['CLARIFAI_PAT'],
)

response = client.chat.completions.create(
    model="https://clarifai.com/{user_id}/{app_id}/models/{model_id}",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    }],
    temperature=0.7,
    max_tokens=1024,
    stream=True,
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Clarifai SDK Predict

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/{model_id}")
result = model.predict(prompt="Hello, Good morning!")
print(result)
```

### Clarifai SDK Generate (Streaming)

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/{model_id}")
for chunk in model.generate(prompt="Explain quantum computing in simple terms."):
    print(chunk, end='', flush=True)
```

### Multimodal with Clarifai SDK

```python
from clarifai.client import Model
from clarifai.runners.utils.data_types import Image

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/{model_id}")

response = model.predict(
    prompt="Describe this image.",
    image=Image(url="https://samples.clarifai.com/metro-north.jpg"),
    max_tokens=1024,
    temperature=0.5,
)
print(response)
```

---

## Model Structure

```
ollama-model-upload/
├── 1/
│   └── model.py          # Ollama model implementation
├── config.yaml           # Model configuration
└── requirements.txt      # Dependencies
```

You can edit `model.py` to customize the model behavior.

## References

- [Clarifai Python SDK](https://github.com/Clarifai/clarifai-python)
- [Clarifai Docs](https://docs.clarifai.com/)
- [Ollama](https://ollama.com/)
- [Ollama Python](https://github.com/ollama/ollama-python)
