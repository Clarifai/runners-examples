---
complexity: beginner
framework: fastmcp
model_size: N/A
gpu_required: false
min_gpu_memory: N/A
features: [mcp, browser, search]
model_class: StdioMCPModelClass
task: mcp-server
---


# Browser MCP Server (DuckDuckGo)

This repository demonstrates how to deploy any open-source Model Context Protocol (MCP) server to Clarifai as an API endpoint. The Browser MCP server (DuckDuckGo) provides tools for web search, browsing, and information retrieval.

---

## Overview

You can upload any open-source MCP server to Clarifai as an API by simply adding the `mcp_server` configuration in `config.yaml`. This enables you to:

- Expose MCP servers as HTTP APIs accessible via Clarifai
- Use FastMCP client to interact with MCP servers
- Easily integrate MCP tools with LLMs for enhanced capabilities

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
  id: "browser-mcp-server"        # You can change this to your preferred model ID
  user_id: "YOUR_USER_ID"         # Replace with your Clarifai user ID
  app_id: "YOUR_APP_ID"           # Replace with your Clarifai app ID
  model_type_id: "text-to-text"

mcp_server:
  command: "uvx"
  args: ["duckduckgo-mcp-server"]
```

The `mcp_server` section specifies:
- **command**: The command to run the MCP server (e.g., `npx`, `uvx`, `python`)
- **args**: Arguments passed to the command to start the MCP server

#### Compute Requirements

The model requires the following compute resources as specified in `config.yaml`:

- CPU: 1000m (1 core)
- CPU Memory: 1Gi
- GPU: Not required (0 accelerators)

Make sure your Clarifai compute cluster and nodepool meets these requirements before deploying the model.

### 2. Upload the Model

Use the Clarifai CLI to upload your model:

```bash
clarifai model upload . --skip_dockerfile
```
Make sure to use `--skip_dockerfile` flag while uploading the model 

This command will:
- Build a Docker image provided in the model folder
- Upload the model to your Clarifai account
- Make it available for inference via HTTP API

**Note**: The DuckDuckGo MCP server does not require any authentication tokens or secrets, making it easy to deploy and use.

---

## Using with FastMCP Client

Once deployed, you can interact with the MCP server using the FastMCP client:

```python
import asyncio
import os
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="https://api.clarifai.com/v2/ext/mcp/v1/users/{user_id}/apps/{app_id}/models/{model_id}",
    headers={"Authorization": "Bearer " + os.environ["CLARIFAI_PAT"]},
)

async def main():
    try:
        async with Client(transport) as client:
            # List all available tools
            tools = await client.list_tools()
            print(f"Available tools: {tools}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

### Available Tools

The DuckDuckGo MCP server provides various tools for web search and information retrieval, such as:
- `ddg_search` - Search the web using DuckDuckGo
- And more...

Use `client.list_tools()` to see all available tools for your specific MCP server.

---

## Integration with LLMs

MCP servers can be easily integrated with LLMs to provide enhanced capabilities. The tools discovered from the MCP server can be used as function calling tools in LLM conversations.

### Example: Using with OpenAI-Compatible API

```python
import asyncio
import os
import json
from openai import AsyncOpenAI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Setup MCP client
transport = StreamableHttpTransport(
    url="https://api.clarifai.com/v2/ext/mcp/v1/users/{user_id}/apps/{app_id}/models/{model_id}",
    headers={"Authorization": f"Bearer {os.environ['CLARIFAI_PAT']}"},
)

# Setup OpenAI client
openai_client = AsyncOpenAI(
    api_key=os.environ["CLARIFAI_PAT"],
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)

def format_tools_to_openai_function(tools):
    """Convert MCP tools to OpenAI function format"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": f"[{tool.name}] {tool.description}",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]

async def main():
    # Get tools from MCP server
    async with Client(transport) as client:
        tools_raw = await client.list_tools()
    
    tools = format_tools_to_openai_function(tools_raw)
    
    # Use tools with LLM
    response = await openai_client.chat.completions.create(
        model="https://clarifai.com/{user_id}/{app_id}/models/{llm_model_id}",
        messages=[
            {"role": "user", "content": "What are the latest developments in artificial intelligence?"}
        ],
        tools=tools,
        tool_choice="auto",
    )
    
    # Handle tool calls and execute them via MCP client
    # ... (implementation details)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## About StdioMCPModelClass

The Browser MCP server uses `StdioMCPModelClass`, which provides powerful features for stdio-based MCP servers:

### Implementation

The model class is defined in `1/model.py`:

```python
from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass

class BrowserMCPServerClass(StdioMCPModelClass):
    pass
```

The `StdioMCPModelClass` handles all the complexity of managing stdio MCP servers, so you only need to inherit from it and configure the `mcp_server` section in `config.yaml`.

---

## Adding Other MCP Servers

To deploy a different MCP server, simply update the `mcp_server` section in `config.yaml`:

### Example: GitHub MCP Server

```yaml
mcp_server:
  command: "npx"
  args: ["-y", "@modelcontextprotocol/server-github"]
```

### Example: Custom Python MCP Server

```yaml
mcp_server:
  command: "python"
  args: ["-m", "my_mcp_server"]
```

The `StdioMCPModelClass` will automatically:
1. Start the stdio process
2. Discover all available tools
3. Expose them via HTTP API
4. Handle tool execution and responses

---

## Requirements

See `requirements.txt` for Python dependencies:

- `clarifai` - Clarifai Python SDK
- `anyio` - Async I/O library
- `mcp` - Model Context Protocol library
- `fastmcp` - FastMCP framework
- `requests` - HTTP library

---

## Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Clarifai Documentation](https://docs.clarifai.com/)


