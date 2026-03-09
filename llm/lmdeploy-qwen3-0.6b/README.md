# Qwen3-0.6B (LMDeploy)

Qwen3-0.6B served via LMDeploy. A compact 0.6B parameter model from the Qwen team, ideal for lightweight inference and edge deployment scenarios.

---

## Quickstart

### 1. Install the Clarifai Python SDK

```bash
pip install clarifai
```

### 2. Login to Clarifai

```bash
clarifai login
```

### 3. Deploy

```bash
clarifai model deploy llm/lmdeploy-qwen3-0.6b
```

---

## OpenAI-Compatible API Usage

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["CLARIFAI_PAT"],
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)

response = client.chat.completions.create(
    model="https://clarifai.com/{user_id}/{app_id}/models/qwen3-0_6b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain quantum entanglement?"}
    ],
    max_completion_tokens=100,
    temperature=0.7,
    stream=True,
)
```

---

## Predict with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/qwen3-0_6b")
result = model.predict(prompt="What are the key differences between classical and quantum computing?")
print(result)
```

## Streaming with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/qwen3-0_6b")
for chunk in model.generate(prompt="Discuss the implications of AI in modern healthcare."):
    print(chunk, end='', flush=True)
```
