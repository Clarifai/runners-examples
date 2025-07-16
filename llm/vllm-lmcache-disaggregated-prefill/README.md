# Llama 3.2-1B Model
This Model uses LMCache and vLLM for efficient LLM inference 
### what is lmcache?

LMCache is an LLM serving engine extension designed to improve the performance of Large Language Models (LLMs), particularly in scenarios involving long contexts. It essentially functions as a high-performance KV cache layer for LLM inference.

In particular it uses `LMCache's KV Cache Offloading` that  moves the Key-Value (KV) caches from the GPU's limited and expensive memory to more abundant and cost-effective storage locations like CPU DRAM or even disk.

### What KV Cache Offloading Does?
- **Relieves GPU Memory Pressure:** KV caches can consume substantial amounts of GPU memory, especially with long contexts or larger models. By offloading these caches to CPU or disk, LMCache frees up valuable GPU memory. This enables the deployment of larger models or the handling of more concurrent requests on the same GPU hardware.

- **Enables More KV Cache Hits:** Limited GPU memory restricts the amount of KV cache that can be stored. Offloading allows for a much larger overall KV cache. This significantly increases the probability of finding an already computed KV cache for a given text segment, leading to a higher rate of "cache hits" and reduced recomputation.

- **Supports Long-Context Scenarios:** In use cases involving multi-round conversations or RAG, where prompts and contexts are frequently reused, offloading is critical. It allows LMCache to store and reuse relevant KV caches efficiently, drastically reducing the Time to First Token (TTFT) and improving overall throughput.
---

## Quickstart

### 1\. Install the Clarifai Python SDK

```bash
pip install clarifai
```

### 2\. Run the `clarifai login` command to login to the Clarifai platform 

```bash
clarifai login
```
After running the login command, you'll be prompted to enter the following details to authenticate your connection:

```
context name (default: "default"):
user id:
personal access token value (default: "ENVVAR" to get our env var rather than config):
```
* **Context name** — You can provide a custom name for your Clarifai configuration context, or simply press Enter to use the default name, "default". This helps you manage different configurations if needed.
* **User ID** — Enter your Clarifai user ID.
* **PAT** — Enter your Clarifai [PAT](https://docs.clarifai.com/compute/models/upload/local-runners#get-a-pat-key).
Retrieve your PAT from your Clarifai account [security settings](https://clarifai.com/settings/security).

---

## Model Upload Guide

### 1. Update Configuration

Before uploading, update the `config.yaml` file with your Clarifai credentials:

```yaml
model:
  id: "vllm-lmcache-kvcache-offloading-llama-3_2-1b"  # You can change this to your preferred model ID
  user_id: "YOUR_USER_ID"      # Replace with your Clarifai user ID
  app_id: "YOUR_APP_ID"        # Replace with your Clarifai app ID
  model_type_id: "text-to-text"
```

---

#### Compute Requirements

The model requires the following compute resources as specified in `config.yaml`:

- CPU: 3 core
- CPU Memory: 12Gi
- GPU: 1 NVIDIA GPU (any type)
- GPU Memory: 19Gi

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

This Model is accessible via the OpenAI-compatible API endpoint. You can utilize the OpenAI Python package as follows:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["CLARIFAI_PAT"],  # Your Clarifai PAT key
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