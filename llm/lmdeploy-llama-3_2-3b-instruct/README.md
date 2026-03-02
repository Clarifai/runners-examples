---
complexity: intermediate
framework: lmdeploy
model_size: 3B
gpu_required: true
min_gpu_memory: 20Gi
features: [text-generation, chat, streaming]
model_class: OpenAIModelClass
task: text-generation
---


# Llama 3.2 Model

The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They outperform many of the available open source and closed chat models on common industry benchmarks.

---

## Quickstart

### 1\. Install the Clarifai Python SDK

```bash
pip install clarifai
```

### 2\. Set your Clarifai Personal Access Token (PAT)

Retrieve your PAT from your Clarifai account security settings.

```bash
export CLARIFAI_PAT="your_personal_access_token"
```

---

## Model Upload Guide

### 1. Update Configuration

Before uploading, update the `config.yaml` file with your Clarifai credentials:

```yaml
model:
  id: "Llama-3_2-3B-Instruct"  # You can change this to your preferred model ID
  user_id: "YOUR_USER_ID"      # Replace with your Clarifai user ID
  app_id: "YOUR_APP_ID"        # Replace with your Clarifai app ID
  model_type_id: "text-to-text"
```

---

#### Compute Requirements

The model requires the following compute resources as specified in `config.yaml`:

- CPU: 1 core
- CPU Memory: 6Gi
- GPU: 1 NVIDIA GPU (any type)
- GPU Memory: 20Gi

Make sure your Clarifai compute cluster and nodepool meets these requirements before deploying the model.

### 2. Upload the Model

Use the Clarifai CLI to upload your model:

```bash
clarifai model upload .
```

This command will:
- Build a Docker image based on your configuration
- Upload the model to your Clarifai account
- Make it available for inference

---

## OpenAI-Compatible API Usage

Llama-3.2 is accessible via the OpenAI-compatible API endpoint. You can utilize the OpenAI Python package as follows:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["CLARIFAI_PAT"],  # Your Clarifai API key
    base_url="https://api.clarifai.com/v2/ext/openai/v1"  # Clarifai's OpenAI-compatible API endpoint
)

response = client.chat.completions.create(
    model="https://clarifai.com/{user_id}/{appid}/models/{model_id}",  # Clarifai model URL
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain the concept of quantum entanglement?"}
    ],
    max_completion_tokens=100,
    temperature=0.7,
    stream=True,
)
```

---

## Predict with Text with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{appid}/models/{model_id}")
prompt = "What are the key differences between classical and quantum computing?"
result = model.predict(prompt)
print("Predict response:", result)
```

---

## Streaming/Generate with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{appid}/models/{model_id}")
for chunk in model.generate(prompt="Discuss the implications of AI in modern healthcare."):
    print(chunk, end='', flush=True)
```