# SmolLM2-1.7B-Instruct Model

SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device. More details in our paper: https://arxiv.org/abs/2502.02737v1

The SmolLM2-1.7B-Instruct demonstrates significant advances over its predecessor SmolLM1-1.7B, particularly in instruction following, knowledge, reasoning, and mathematics. It was trained on 11 trillion tokens using a diverse dataset combination

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
  id: "SmolLM2-1_7B-Instruct"  # You can change this to your preferred model ID
  user_id: "YOUR_USER_ID"      # Replace with your Clarifai user ID
  app_id: "YOUR_APP_ID"        # Replace with your Clarifai app ID
  model_type_id: "text-to-text"
```

---

#### Compute Requirements

The model requires the following compute resources as specified in `config.yaml`:

- CPU: 1 core
- CPU Memory: 6GB
- GPU: 1 NVIDIA GPU (any type)
- GPU Memory: 20GB

Make sure your Clarifai compute cluster meets these requirements before deploying the model.

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

SmolLM2-1.7B is accessible via the OpenAI-compatible API endpoint. You can utilize the OpenAI Python package as follows:

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