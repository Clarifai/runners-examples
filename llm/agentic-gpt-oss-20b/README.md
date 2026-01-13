# Agentic GPT-OSS-20B Model

The GPT-OSS-20B model with agentic capabilities enabled through MCP (Model Context Protocol) server integration. This model extends `OpenAIModelClass` with `AgenticModelClass` to provide tool discovery, execution, and iterative tool calling capabilities for both chat completions and responses endpoints, supporting both streaming and non-streaming modes.

---

## Quickstart

### 1. Install the Clarifai Python SDK

```bash
pip install clarifai
```

### 2. Set your Clarifai Personal Access Token (PAT)

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
  id: "gpt-oss-20b"  # You can change this to your preferred model ID
  user_id: "YOUR_USER_ID"      # Replace with your Clarifai user ID
  app_id: "YOUR_APP_ID"        # Replace with your Clarifai app ID
  model_type_id: "text-to-text"
```

### 2. Compute Requirements

The model requires the following compute resources as specified in `config.yaml`:

- CPU: 8 cores
- CPU Memory: 30Gi (limit), 20Gi (requests)
- GPU: 1 NVIDIA GPU (any type)
- GPU Memory: 48Gi

Make sure your Clarifai compute cluster and nodepool meets these requirements before deploying the model.

### 3. Upload the Model

Use the Clarifai CLI to upload your model:

```bash
clarifai model upload .
```

This command will:
- Build a Docker image based on your configuration
- Upload the model to your Clarifai account
- Make it available for inference

---

## Agentic Behavior with MCP Servers

The GPT-OSS-20B model supports agentic behavior through MCP (Model Context Protocol) servers. MCP servers provide tools that the model can discover and use autonomously to complete tasks.

### LLM with MCP Servers Client Example

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ['CLARIFAI_PAT']
)

# Define MCP servers to use
mcp_servers = [
    "https://clarifai.com/clarifai/mcp/models/mcp-server-weather",
    "https://clarifai.com/clarifai/mcp/models/browser-mcp-server",
    "https://clarifai.com/clarifai/mcp/models/time-mcp-server",
]

# Create a chat completion with MCP servers
completion = client.chat.completions.create(
    model="https://clarifai.com/clarifai/agentic-model/models/gpt-oss-20b",
    messages=[{"role": "user", "content": "What was the weather in Los Angeles, California yesterday?"}],
    extra_body={"mcp_servers": mcp_servers},
    max_completion_tokens=10000,
    stream=True
)

# Stream the response
for chunk in completion:
    if chunk.choices and len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    elif chunk.choices and len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
```

### Non-Streaming Example

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ['CLARIFAI_PAT']
)

mcp_servers = [
    "https://clarifai.com/clarifai/mcp/models/mcp-server-weather",
    "https://clarifai.com/clarifai/mcp/models/browser-mcp-server",
]

completion = client.chat.completions.create(
    model="https://clarifai.com/clarifai/agentic-model/models/gpt-oss-20b",
    messages=[{"role": "user", "content": "What's the current time in New York and what's the weather there?"}],
    extra_body={"mcp_servers": mcp_servers},
    max_completion_tokens=10000,
    stream=False
)

print(completion.choices[0].message.content)
```

---

## Available MCP Servers

The model can work with various MCP servers. Some examples include:

- **Weather Server**: `https://clarifai.com/clarifai/mcp/models/mcp-server-weather` - Provides weather information
- **Browser Server**: `https://clarifai.com/clarifai/mcp/models/browser-mcp-server` - Enables web browsing capabilities
- **Time Server**: `https://clarifai.com/clarifai/mcp/models/time-mcp-server` - Provides time and date information

You can specify multiple MCP servers in the `mcp_servers` list to give the model access to multiple tool sets.

---

## OpenAI-Compatible API Usage

The GPT-OSS-20B model is accessible via the OpenAI-compatible API endpoint. You can utilize the OpenAI Python package as shown in the examples above.

### Basic Usage (Without MCP Servers)

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["CLARIFAI_PAT"],
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)

response = client.chat.completions.create(
    model="https://clarifai.com/{user_id}/{app_id}/models/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_completion_tokens=1000,
    temperature=0.7,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Predict with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/gpt-oss-20b")

# Predict with MCP servers
result = model.predict(
    prompt="What was the weather in San Francisco yesterday?",
    mcp_servers=[
        "https://clarifai.com/clarifai/mcp/models/mcp-server-weather",
        "https://clarifai.com/clarifai/mcp/models/time-mcp-server"
    ]
)
print("Predict response:", result)
```

---

## Streaming/Generate with Clarifai SDK

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/{user_id}/{app_id}/models/gpt-oss-20b")

# Generate with MCP servers (streaming)
for chunk in model.generate(
    prompt="Browse the web and tell me the latest news about AI.",
    mcp_servers=[
        "https://clarifai.com/clarifai/mcp/models/browser-mcp-server"
    ]
):
    print(chunk, end='', flush=True)
```

---

## Model Details

- **Base Model**: OpenAI GPT-OSS-20B
- **Framework**: vLLM
- **Model Type**: Text-to-Text
- **Agentic Capabilities**: Yes (via MCP servers)
- **Streaming Support**: Yes
- **Tool Calling**: Yes (via MCP protocol)

---

## Notes

- The model automatically discovers tools from the specified MCP servers
- The model can iteratively call tools to complete complex tasks
- Both streaming and non-streaming modes are supported
- The model supports reasoning content in streaming mode (check for `reasoning_content` in delta)

