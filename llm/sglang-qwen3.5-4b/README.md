# Qwen3.5-4B (SGLang)

Qwen3.5-4B served via SGLang. A 4B parameter instruction-tuned model from the Qwen team, balancing capability and efficiency.

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

### 3. Test Locally

```bash
clarifai model serve llm/sglang-qwen3.5-4b
```

### 4. Deploy

```bash
clarifai model deploy llm/sglang-qwen3.5-4b
```

---

## Predict with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/qwen35-4b")
result = model.predict(prompt="Hello, how are you?")
print(result)
```

## Streaming with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/qwen35-4b")
for chunk in model.generate(prompt="Explain quantum computing in simple terms."):
    print(chunk, end='', flush=True)
```
