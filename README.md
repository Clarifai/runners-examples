![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)
# Clarifai Model Deployment

[Clarifai](https://www.clarifai.com/) provides an easy-to-use platform to serve AI/ML models in production.

This guide walks you through deploying models on Clarifai — from scaffolding a project to running predictions in production.

## Quick Start

Three commands to go from zero to a deployed model:

```bash
pip install -U clarifai

# 1. Authenticate
clarifai login

# 2. Scaffold a model (auto-selects GPU based on model size)
clarifai model init --toolkit vllm --model-name Qwen/Qwen3-0.6B

# 3. Deploy to production (auto-creates all infrastructure)
clarifai model deploy ./Qwen3-0.6B
```

The CLI handles compute cluster creation, nodepool setup, image building, and deployment automatically. When it finishes you'll see:

```
── Ready ──────────────────────────────────────────────
  Model deployed successfully!

  Model:       https://clarifai.com/you/main/models/qwen3-0-6b
  Deployment:  deploy-qwen3-0-6b-abc123
  Instance:    g5.xlarge

── Next Steps ─────────────────────────────────────────
  Predict:     clarifai model predict you/main/models/qwen3-0-6b "Hello"
  Logs:        clarifai model logs --deployment "deploy-qwen3-0-6b-abc123"
  Undeploy:    clarifai model undeploy --deployment "deploy-qwen3-0-6b-abc123"
```

### Which path is right for you?

| I want to... | Do this |
|--------------|---------|
| **Deploy a HuggingFace LLM** (Qwen, Llama, Gemma, etc.) | `clarifai model init --toolkit vllm --model-name <hf-repo>` then `clarifai model deploy` |
| **Deploy an MCP tool server** | `clarifai model init --toolkit mcp` then write your tools |
| **Wrap an existing API or local server** | `clarifai model init --toolkit openai` or `--toolkit ollama` |
| **Build a fully custom model** (vision, embeddings, etc.) | `clarifai model init` then edit model.py — see [Writing a Custom Model](#writing-a-custom-model) |

## Contents

- [Quick Start](#quick-start)
- [Installation & Authentication](#installation--authentication)
- [Model Development Workflow](#model-development-workflow)
  - [Step 1: Scaffold a Model](#step-1-scaffold-a-model-clarifai-model-init)
  - [Step 2: Test Locally](#step-2-test-locally-clarifai-model-serve)
  - [Step 3: Deploy to Production](#step-3-deploy-to-production-clarifai-model-deploy)
  - [Step 4: Run Predictions](#step-4-run-predictions-clarifai-model-predict)
  - [Step 5: Manage Your Deployment](#step-5-manage-your-deployment)
- [Writing a Custom Model](#writing-a-custom-model)
  - [Model Folder Structure](#model-folder-structure)
  - [config.yaml](#configyaml)
  - [model.py](#modelpy)
  - [requirements.txt](#requirementstxt)
- [Model Types](#model-types)
  - [Python Model (ModelClass)](#python-model-modelclass)
  - [LLM / OpenAI-Compatible Model (OpenAIModelClass)](#llm--openai-compatible-model-openaimodelclass)
  - [MCP Tool Server (MCPModelClass)](#mcp-tool-server-mcpmodelclass)
- [Using the Python SDK Client](#using-the-python-sdk-client)
- [Available Examples](#available-examples)
- [Supported Data Types](#supported-data-types)
- [Advanced Topics](#advanced-topics)

---

## Installation & Authentication

### Install the SDK

```bash
pip install -U clarifai
```

### Authenticate

```bash
# Interactive (prompts for your PAT)
clarifai login

# Non-interactive
clarifai login --pat $MY_PAT

# Org account
clarifai login --pat $PAT --user-id my-org

# Dev environment
clarifai login https://api-dev.clarifai.com
```

The CLI saves your credentials locally. Verify with:

```bash
clarifai whoami
```

You can manage multiple environments (prod, staging, org accounts) using named contexts:

```bash
clarifai config ls                  # List all contexts
clarifai config use <name>          # Switch active context
```

> **Note:** You can generate a PAT in your Clarifai account under [Personal Settings → Security](https://clarifai.com/settings/security).

---

## Model Development Workflow

```
init  →  serve (optional)  →  deploy  →  predict  →  manage
scaffold   test locally       push to      run         status / logs /
project    before deploying   production   inference   undeploy
```

### Step 1: Scaffold a Model (`clarifai model init`)

Generate a ready-to-deploy model project with a single command:

```bash
# Deploy a HuggingFace LLM with vLLM
clarifai model init --toolkit vllm --model-name Qwen/Qwen3-0.6B

# Deploy with SGLang
clarifai model init --toolkit sglang --model-name Qwen/Qwen2-7B

# Deploy a HuggingFace model directly
clarifai model init --toolkit huggingface --model-name google/gemma-2b

# Wrap a local Ollama model
clarifai model init --toolkit ollama --model-name llama3.1

# Create an MCP tool server
clarifai model init --toolkit mcp my-mcp-server

# Wrap an OpenAI-compatible API
clarifai model init --toolkit openai my-wrapper

# Blank Python model (full control)
clarifai model init my-model
```

**Available toolkits:**

| Toolkit | Use Case |
|---------|----------|
| `vllm` | High-throughput LLM serving with vLLM |
| `sglang` | Fast LLM serving with SGLang |
| `huggingface` | HuggingFace Transformers (direct inference) |
| `ollama` | Wrap a local Ollama model |
| `lmstudio` | Wrap a local LM Studio model |
| `mcp` | MCP tool server (FastMCP) |
| `openai` | OpenAI-compatible API wrapper |
| `python` | Blank Python model (default) |

**What it creates:**

```
my-model/
├── config.yaml          # Simplified config (auto-filled at deploy time)
├── requirements.txt     # Dependencies
└── 1/
    └── model.py         # Model implementation
```

**Smart GPU selection:** For HuggingFace models, the CLI queries the model's metadata (parameter count, quantization, architecture) and auto-selects the smallest GPU instance that fits:

```
$ clarifai model init --toolkit vllm --model-name Qwen/Qwen3-4B
  Instance: g5.xlarge (Estimated 15.9 GiB VRAM, fits g5.xlarge (22 GiB))
```

---

### Step 2: Test Locally (`clarifai model serve`)

Before deploying, test your model locally:

```bash
# Run in current Python environment (fastest)
clarifai model serve ./my-model

# Auto-create virtualenv and install deps
clarifai model serve ./my-model --mode env

# Build and run inside Docker (recommended for production parity)
clarifai model serve ./my-model --mode container

# Standalone gRPC server (no login required, offline development)
clarifai model serve ./my-model --grpc --port 9000
```

**What happens (default API-connected mode):**
1. Validates your `config.yaml`
2. Starts the model server locally
3. Registers with Clarifai so you can test via the Playground UI
4. Displays a predict command and Playground URL
5. Cleans up on Ctrl+C

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_PATH` | `.` | Model directory containing config.yaml |
| `--mode` | `none` | `none` (current env), `env` (auto-venv), `container` (Docker) |
| `--grpc` | off | Standalone gRPC server, no API connection needed |
| `--port` | 8000 | Server port (with `--grpc`) |
| `--concurrency` | 32 | Max concurrent requests |
| `--keep-image` | off | Keep Docker image after exit (container mode) |

---

### Step 3: Deploy to Production (`clarifai model deploy`)

Deploy your model to Clarifai cloud compute in one command. All infrastructure (compute cluster, nodepool, deployment) is created automatically:

```bash
# Deploy (uses instance from config.yaml)
clarifai model deploy ./my-model

# Override instance at deploy time
clarifai model deploy ./my-model --instance g6e.2xlarge

# GPU shorthands work too
clarifai model deploy ./my-model --instance a10g

# Specific cloud and region
clarifai model deploy ./my-model --instance g5.xlarge --cloud aws --region us-west-2

# Autoscaling
clarifai model deploy ./my-model --min-replicas 2 --max-replicas 10

# Deploy an already-uploaded model by URL
clarifai model deploy --model-url https://clarifai.com/user/app/models/id --instance g5.xlarge
```

**Browse available instances:**

```bash
$ clarifai list-instances

  Instance          Cloud   Region       GPUs  Accelerator  GPU Mem
  ──────────────────────────────────────────────────────────────────
  g5.xlarge         AWS     us-east-1    1     NVIDIA-A10G  24 GiB
  g6e.2xlarge       AWS     us-east-1    1     NVIDIA-L40S  48 GiB
  t3a.2xlarge       AWS     us-east-1    0     CPU          -
  n1-standard-4-t4  GCP     us-central1  1     NVIDIA-T4    16 GiB
  ...
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_PATH` | `.` | Local model directory |
| `--instance` | auto | Hardware instance (e.g., `g5.xlarge`, `a10g`) |
| `--model-url` | - | Deploy an already-uploaded model (skip upload) |
| `--min-replicas` | 1 | Minimum running replicas |
| `--max-replicas` | 5 | Maximum replicas for autoscaling |
| `--cloud` | auto | Cloud provider (`aws`, `gcp`, `vultr`) |
| `--region` | auto | Cloud region (e.g., `us-east-1`) |

**Deploy phases:**

```
── Validate ──  Config validation, HuggingFace repo access check
── Upload ────  Build and push model image
── Deploy ────  Create compute cluster, nodepool, deployment
── Monitor ───  Stream pod events until ready
── Ready ─────  Show model URL, predict command, status/log commands
```

---

### Step 4: Run Predictions (`clarifai model predict`)

Once deployed, run predictions directly from the CLI:

```bash
# Text input
clarifai model predict user/app/models/my-model "Hello world"

# Pipe from stdin
echo "Explain quantum computing" | clarifai model predict user/app/models/my-model

# Image/video/audio file
clarifai model predict user/app/models/detector --file photo.jpg

# Image URL
clarifai model predict user/app/models/detector --url https://example.com/img.jpg

# Named parameters
clarifai model predict user/app/models/llm -i prompt="Hello" -i max_tokens=200

# OpenAI chat mode
clarifai model predict openai/chat-completion/models/GPT-4 --chat "What is AI?"

# JSON output
clarifai model predict user/app/models/my-model "Hello" -o json

# Inspect available methods and their parameters
clarifai model predict user/app/models/my-model --info
```

| Option | Description |
|--------|-------------|
| `MODEL` | Model as `user/app/models/id` or full URL |
| `INPUT` | Text input (positional) |
| `--file PATH` | Local file (image, audio, video) |
| `--url URL` | Remote media URL |
| `-i KEY=VALUE` | Named parameter (repeatable) |
| `--chat TEXT` | OpenAI chat mode |
| `--info` | Show available methods, then exit |
| `-o text\|json` | Output format (default: `text`) |
| `--deployment ID` | Route to a specific deployment |

---

### Step 5: Manage Your Deployment

**Check status:**

```bash
clarifai model status --deployment deploy-abc123
clarifai model status user/app/models/my-model
```

**Stream logs:**

```bash
# Model stdout/stderr
clarifai model logs --deployment deploy-abc123

# Kubernetes scheduling/scaling events
clarifai model logs --deployment deploy-abc123 --log-type events

# Print current logs and exit
clarifai model logs --deployment deploy-abc123 --no-follow
```

**Remove a deployment:**

```bash
clarifai model undeploy --deployment deploy-abc123
```

---

## Writing a Custom Model

> **Most users don't need this section.** If you're deploying a HuggingFace model, `clarifai model init --toolkit vllm` generates everything for you. Read on if you want to customize the generated code or build a model from scratch.

### Model Folder Structure

```
my-model/
├── config.yaml          # Model configuration
├── requirements.txt     # Python dependencies
└── 1/
    └── model.py         # Model implementation
```

### config.yaml

The simplified config requires only a few fields. Everything else is auto-filled at deploy time:

```yaml
model:
  id: "my-model"

build_info:
  python_version: "3.11"

compute:
  instance: g5.xlarge

checkpoints:
  repo_id: Qwen/Qwen3-0.6B
  type: huggingface
  when: runtime
```

**Key fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `model.id` | Yes | Model identifier |
| `model.model_type_id` | No | Defaults to `any-to-any` |
| `compute.instance` | Yes (for GPU) | Instance type (run `clarifai list-instances` to see options) |
| `build_info.python_version` | No | Defaults to `3.12` |
| `checkpoints.repo_id` | No | HuggingFace repo for auto-download |
| `checkpoints.when` | No | `runtime` (default), `build`, or `upload` |

> **Tip:** `user_id` and `app_id` are resolved automatically from your CLI context — you don't need to put them in config.yaml.

> **Tip:** For large models, use `when: runtime` to download checkpoints at runtime. This keeps Docker images small and speeds up builds.

**Using a custom Docker base image** (for vLLM, SGLang, etc.):

```yaml
build_info:
  image: "lmsysorg/sglang:latest"

checkpoints:
  repo_id: Qwen/Qwen3-4B
  type: huggingface
  when: runtime

compute:
  instance: g5.xlarge

model:
  id: qwen3-4b
```

### model.py

Your model must define a class that inherits from one of the base classes (`ModelClass`, `OpenAIModelClass`, or `MCPModelClass`) and expose at least one method decorated with `@<BaseClass>.method`.

```python
from typing import Iterator
from clarifai.runners.models.model_class import ModelClass

class MyModel(ModelClass):

    def load_model(self):
        """Called once at startup. Load weights, initialize pipelines, etc."""
        self.model = ...

    @ModelClass.method
    def predict(self, prompt: str = "", max_tokens: int = 256) -> str:
        """Single request → single response."""
        return self.model(prompt, max_tokens=max_tokens)

    @ModelClass.method
    def generate(self, prompt: str = "") -> Iterator[str]:
        """Single request → streamed response."""
        for token in self.model.stream(prompt):
            yield token
```

**Method types** (determined by type hints):

| Pattern | Signature | Use Case |
|---------|-----------|----------|
| Unary | `def predict(self, ...) -> str` | Standard request-response |
| Server streaming | `def generate(self, ...) -> Iterator[str]` | LLM token streaming |
| Bidirectional streaming | `def stream(self, input: Iterator[str]) -> Iterator[str]` | Real-time processing |

**`load_model()` (optional):** Use this for expensive one-time initialization — loading model weights, creating pipelines, downloading checkpoints. Called once when the container starts.

**Testing:** When you run `clarifai model serve`, it loads your model and you can test via the Playground UI or CLI predict.

### requirements.txt

List all Python dependencies your model needs:

```
torch==2.5.1
transformers
accelerate
```

> **Tip:** If your model requires PyTorch, specify the version explicitly (e.g., `torch==2.5.1` or `torch==2.4.1`). Clarifai provides optimized pre-built images for these versions that significantly speed up builds and reduce cold start times.

---

## Model Types

Clarifai provides three base classes. Pick the one that matches your use case:

| Base Class | When to Use |
|------------|-------------|
| `ModelClass` | Custom Python inference — classification, detection, embeddings, any logic |
| `OpenAIModelClass` | LLM serving via OpenAI-compatible API (vLLM, SGLang, LMDeploy) |
| `MCPModelClass` | MCP tool servers for AI agents (FastMCP) |

### Python Model (ModelClass)

For custom inference logic — classification, detection, embeddings, or any Python code.

```python
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Image, Text

class ImageCaptioner(ModelClass):

    def load_model(self):
        from transformers import pipeline
        self.pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    @ModelClass.method
    def predict(self, image: Image) -> str:
        pil_image = image.to_pil()
        result = self.pipe(pil_image)
        return result[0]["generated_text"]
```

**See:** [`hello-world/`](hello-world/) for a minimal example.

---

### LLM / OpenAI-Compatible Model (OpenAIModelClass)

For models served via an OpenAI-compatible API (vLLM, SGLang, LMDeploy, etc.):

```python
from typing import Iterator, List
from openai import OpenAI
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages

class MyLLM(OpenAIModelClass):
    client = OpenAI(api_key="local-key", base_url="http://localhost:8000/v1")
    model = client.models.list().data[0].id

    def load_model(self):
        """Start the inference server (vLLM, SGLang, etc.)."""
        pass

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=256, description="Max tokens to generate."),
        temperature: float = Param(default=1.0, description="Sampling temperature."),
    ) -> str:
        messages = build_openai_messages(prompt, chat_history)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, temperature=temperature,
        )
        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=256, description="Max tokens to generate."),
        temperature: float = Param(default=1.0, description="Sampling temperature."),
    ) -> Iterator[str]:
        messages = build_openai_messages(prompt, chat_history)
        for chunk in self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, temperature=temperature, stream=True,
        ):
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
```

**See:** [`llm/vllm-qwen3-0.6B/`](llm/vllm-qwen3-0.6B/) for a vLLM example, or [`llm/sglang-qwen3.5-4b/`](llm/sglang-qwen3.5-4b/) for SGLang.

> **Tip:** The fastest way to deploy an LLM is `clarifai model init --toolkit vllm --model-name <hf-repo>` — it generates the full model.py, config.yaml, and requirements.txt with the right server setup.

---

### MCP Tool Server (MCPModelClass)

For deploying [MCP](https://modelcontextprotocol.io/) tool servers that AI agents can use:

```python
from fastmcp import FastMCP
from pydantic import Field
from clarifai.runners.models.mcp_class import MCPModelClass

server = FastMCP("my-tools", instructions="Useful tools for agents.", stateless_http=True)

@server.tool("hello", description="Say hello to someone")
def hello(name: str = Field(description="Name to greet")) -> str:
    return f"Hello, {name}!"

@server.resource("config://version")
def get_version():
    return "1.0.0"

class MyModel(MCPModelClass):
    def get_server(self) -> FastMCP:
        return server
```

**See:** [`mcp/math/`](mcp/math/) for a minimal example, or the [`mcp/`](mcp/) directory for real-world MCP servers (GitHub, Postgres, browser tools, code execution, etc.).

---

## Using the Python SDK Client

In addition to the CLI (`clarifai model predict`), you can call your deployed model from Python code. The SDK client mirrors the exact method signatures defined in your `model.py`:

```python
from clarifai.client.model import Model

# Initialize by model URL
model = Model(model_url="https://clarifai.com/user/app/models/my-model")

# Or by IDs
model = Model(user_id="user", app_id="app", model_id="my-model")
```

**Discover methods and signatures:**

```python
# List available methods
print(model.available_methods())

# Get method signature (parameter names, types, defaults)
print(model.method_signature(method_name="predict"))

# Generate a ready-to-use code snippet
print(model.generate_client_script())
```

**Unary prediction (request → response):**

```python
result = model.predict(prompt="Hello world")
print(result)

# Batch prediction (multiple inputs)
results = model.predict([
    {"prompt": "Hello"},
    {"prompt": "World"},
])
```

**Streaming prediction (request → stream of responses):**

```python
for chunk in model.generate(prompt="Explain quantum computing"):
    print(chunk, end="", flush=True)
```

**Bidirectional streaming:**

```python
from clarifai.runners.utils.data_types import Audio

for transcript in model.transcribe(audio=iter(Audio(bytes=b"..."))):
    print(transcript.text)
```

---

## Available Examples

### LLM Models (`llm/`)

| Example | Toolkit | Description |
|---------|---------|-------------|
| [`vllm-qwen3-0.6B`](llm/vllm-qwen3-0.6B/) | vLLM | Qwen3 0.6B with vLLM |
| [`sglang-qwen3.5-4b`](llm/sglang-qwen3.5-4b/) | SGLang | Qwen3.5 4B with SGLang |
| [`lmdeploy-qwen3-0.6b`](llm/lmdeploy-qwen3-0.6b/) | LMDeploy | Qwen3 0.6B with LMDeploy |
| [`hf-llama-3_2-1b-instruct`](llm/hf-llama-3_2-1b-instruct/) | HuggingFace | Llama 3.2 1B with Transformers |
| [`agentic-gpt-5_1`](llm/agentic-gpt-5_1/) | Agentic | Agentic model with tool use |
| [`agentic-gpt-oss-20b`](llm/agentic-gpt-oss-20b/) | Agentic | Open-source agentic model |

### MCP Tool Servers (`mcp/`)

| Example | Description |
|---------|-------------|
| [`browser-tools`](mcp/browser-tools/) | Web browsing, scraping, and search |
| [`github-mcp-server`](mcp/github-mcp-server/) | GitHub operations (issues, PRs, repos) |
| [`postgres`](mcp/postgres/) | PostgreSQL database operations |
| [`google-drive`](mcp/google-drive/) | Google Drive file management |
| [`code-execution-docker-version`](mcp/code-execution-docker-version/) | Sandboxed code execution |
| [`slack-tools-server`](mcp/slack-tools-server/) | Slack integration |
| [`web-search`](mcp/web-search/) | Web search functionality |

### Multimodal Models (`multimodal-models/`)

| Example | Description |
|---------|-------------|
| [`qwen2_5-VL-3B-Instruct-vllm`](multimodal-models/qwen2_5-VL-3B-Instruct-vllm/) | Qwen2.5 Vision 3B with vLLM |
| [`sglang-deepseek-ocr`](multimodal-models/sglang-deepseek-ocr/) | DeepSeek OCR with SGLang |

### Other Models

| Example | Description |
|---------|-------------|
| [`hello-world`](hello-world/) | Minimal starter (text in → text out) |
| [`image-classifier`](image-classifier/) | Image classification |
| [`image-detector`](image-detector/) | Object detection |
| [`image-segmenter`](image-segmenter/) | Image segmentation |
| [`text-embedder`](text-embedder/) | Text embeddings |
| [`ocr`](ocr/) | Optical character recognition |
| [`multimodal-models`](multimodal-models/) | Multi-modal examples |

---

## Supported Data Types

Clarifai's model framework supports rich data typing for both inputs and outputs. Each method parameter and return type must be annotated.

**Common types:**

| Type | Import | Example |
|------|--------|---------|
| `str`, `int`, `float`, `bool` | built-in | `prompt: str = ""` |
| `Text` | `clarifai.runners.utils.data_types` | `Text("hello")`, `Text(url="...")` |
| `Image` | `clarifai.runners.utils.data_types` | `Image(url="...")`, `Image.from_pil(img)` |
| `Audio` | `clarifai.runners.utils.data_types` | `Audio(bytes=b"...")` |
| `Video` | `clarifai.runners.utils.data_types` | `Video(url="...")` |
| `List[T]`, `Dict[K, V]` | `typing` | `List[Image]`, `Dict[str, float]` |
| `Iterator[T]` | `typing` | `Iterator[str]` (for streaming) |

For the full reference including `Concept`, `Region`, `Frame`, `NamedFields`, and container types, see [SUPPORTED_DATATYPE.md](SUPPORTED_DATATYPE.md).

---

## Advanced Topics

### HuggingFace Token for Gated Models

Models like Llama, Gemma, and other gated repos require an HuggingFace token:

```bash
# Set in environment (recommended)
export HF_TOKEN=hf_...

# Or add to config.yaml
checkpoints:
  repo_id: meta-llama/Llama-3.1-8B-Instruct
  hf_token: hf_...
```

The CLI validates HuggingFace access early (before building the Docker image) and provides actionable error messages if access is denied.

> **Note:** When `HF_TOKEN` is in your environment, the CLI automatically persists it to `config.yaml` during deploy so the container can access it.

### Compute Instance Discovery

Browse all available GPU instances across cloud providers:

```bash
clarifai list-instances                      # All instances
clarifai list-instances --cloud aws          # AWS only
clarifai list-instances --gpu H100           # H100 instances
clarifai list-instances --min-gpus 2         # Multi-GPU
clarifai list-instances --min-gpu-mem 48Gi   # 48+ GiB GPU memory
```

### Upload Without Deploying

If you want to upload a model image without immediately deploying it:

```bash
clarifai model upload ./my-model
```

You can deploy it later with:

```bash
clarifai model deploy --model-url https://clarifai.com/user/app/models/my-model --instance g5.xlarge
```

### Model Concepts / Labels

For classification or detection models that output concepts/labels, define them in `config.yaml`:

```yaml
concepts:
  - id: "0"
    name: bus
  - id: "1"
    name: person
  - id: "2"
    name: car
```

> **Note:** If your model loads from HuggingFace with `checkpoints` defined, concepts are inferred automatically.

### Per-Command Context Override

Any command can use a different authentication context for a single invocation:

```bash
clarifai --context staging model deploy ./my-model
clarifai --context prod-openai model predict openai/chat-completion/models/GPT-4 "Hello"
```
