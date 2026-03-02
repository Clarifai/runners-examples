---
complexity: intermediate
framework: fastmcp
model_size: N/A
gpu_required: false
min_gpu_memory: N/A
features: [mcp, github, api-integration]
model_class: StdioMCPModelClass
task: mcp-server
---


# GitHub MCP Server

This repository demonstrates how to deploy any open-source Model Context Protocol (MCP) server to Clarifai as an API endpoint. The GitHub MCP server provides tools for interacting with GitHub repositories, pull requests, issues, and more.

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
  id: "github-mcp-server"        # You can change this to your preferred model ID
  user_id: "YOUR_USER_ID"        # Replace with your Clarifai user ID
  app_id: "YOUR_APP_ID"          # Replace with your Clarifai app ID
  model_type_id: "text-to-text"

mcp_server:
  command: "npx"
  args: ["-y", "@modelcontextprotocol/server-github"]
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

### 2. Configure Secrets (Optional)

This section is not needed if you are hardcoding `GITHUB_PERSONAL_ACCESS_TOKEN` in `mcp_server` like below
```
mcp_server:
  command: "npx"
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_PERSONAL_ACCESS_TOKEN: {GITHUB_PERSONAL_ACCESS_TOKEN}
```

Secrets in clarifai enable you to store and manage sensitive configuration data, such as third-party API keys, without hardcoding them into your scripts

Secrets are securely encrypted values that function as environment variables within your workflows.

Add your GitHub Personal Access Token as a secret in `config.yaml`:

```yaml
secrets:
  - id: "github_personal_access_token"
    type: "env"
    env_var: "GITHUB_PERSONAL_ACCESS_TOKEN"
    description: "GitHub personal access token"
```

Refer: https://docs.clarifai.com/control/authentication/environment-secrets

### 3. Upload the Model

Use the Clarifai CLI to upload your model:

```bash
clarifai model upload .
```

This command will:
- Build a Docker image based on your configuration
- Upload the model to your Clarifai account
- Make it available for inference via HTTP API

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
            
            # Call a tool (example: list pull requests)
            result = await client.call_tool(
                "list_pull_requests", 
                {"owner": "clarifai", "repo": "clarifai-python"}
            )
            print(f"Result: {result[0].text}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

### Available Tools

The GitHub MCP server provides various tools for interacting with GitHub, such as:
- `list_pull_requests` - List pull requests for a repository
- `get_pull_request` - Get details of a specific pull request
- `list_issues` - List issues for a repository
- `get_issue` - Get details of a specific issue
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
            {"role": "user", "content": "List all open pull requests in the clarifai/clarifai-python repository"}
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

The GitHub MCP server uses `StdioMCPModelClass`, which provides powerful features for stdio-based MCP servers:

### Key Features

- **Automatic tool discovery**: Discovers all tools from a **stdio** MCP server at startup and registers them in Python FastMCP
- **Single long-lived Node.js process**: The stdio process (e.g., `npx -y @modelcontextprotocol/server-github`) starts once and stays running
- **Persistent MCP session**: Maintains a single `ClientSession` to the stdio server with double-checked locking for thread safety
- **Dynamic function generation**: Creates properly-typed Python functions for each stdio tool with correct signatures and annotations
- **Configuration via YAML**: Simple setup through `config.yaml` with support for environment variables and secrets

### Implementation

The model class is defined in `1/model.py`:

```python
from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass

class GithubMCPServerClass(StdioMCPModelClass):
    pass
```

The `StdioMCPModelClass` handles all the complexity of managing stdio MCP servers, so you only need to inherit from it and configure the `mcp_server` section in `config.yaml`.

---

## Adding Other MCP Servers

To deploy a different MCP server, simply update the `mcp_server` section in `config.yaml`:

### Example: DuckDuckGo MCP Server

```yaml
mcp_server:
  command: "uvx"
  args: ["duckduckgo-mcp-server"]
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

