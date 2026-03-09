# MCP Server Examples

This directory contains example MCP (Model Context Protocol) servers for deployment on Clarifai. Each example shows how to create or deploy MCP servers that provide tools for LLMs.

## Available Examples

### Stdio MCP Servers (Open Source)

These examples deploy existing open-source MCP servers using `StdioMCPModelClass`:

| Example | Description |
|---------|-------------|
| [browser-mcp-server](browser-mcp-server/) | DuckDuckGo web search (no auth needed) |
| [github-mcp-server](github-mcp-server/) | GitHub repositories, PRs, issues |
| [web-search](web-search/) | Web search capabilities |

### Custom MCP Servers (FastMCP)

These examples implement custom MCP servers using the FastMCP framework:

| Example | Description |
|---------|-------------|
| [browser-tools](browser-tools/) | Web browsing, scraping, and search |
| [firecrawl-browser-tools](firecrawl-browser-tools/) | Advanced web scraping via Firecrawl |
| [slack-tools-server](slack-tools-server/) | Slack messaging and channel management |
| [google-drive](google-drive/) | Google Drive file operations |
| [postgres](postgres/) | PostgreSQL database operations |
| [math](math/) | Mathematical computations |
| [code-execution-docker-version](code-execution-docker-version/) | Python code execution in Docker |
| [code-execution-without-docker-version](code-execution-without-docker-version/) | Python code execution (no Docker) |

## Deploying an MCP Server

### Stdio MCP Servers

For existing open-source MCP servers, just configure `config.yaml`:

```yaml
model:
  id: my-mcp-server
  model_type_id: mcp

mcp_server:
  command: "npx"
  args: ["-y", "@modelcontextprotocol/server-github"]

compute:
  instance: t3a.2xlarge
```

Then deploy:

```bash
clarifai model deploy mcp/github-mcp-server
```

### Custom MCP Servers

For custom servers, implement your tools in `1/model.py`:

```python
from clarifai.runners.models.mcp_class import MCPModelClass
from mcp.server.fastmcp import FastMCP

class MyMCPServer(MCPModelClass):
    def create_mcp_server(self):
        mcp = FastMCP("my-server")

        @mcp.tool()
        def my_tool(query: str) -> str:
            return f"Result for: {query}"

        return mcp
```

Then deploy:

```bash
clarifai model deploy mcp/my-server
```

## Using Deployed MCP Servers

### With FastMCP Client

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
    async with Client(transport) as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        result = await client.call_tool("my_tool", {"query": "hello"})
        print(result[0].text)

asyncio.run(main())
```

### With Agentic LLMs

MCP servers can be attached to LLMs as tool providers:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["CLARIFAI_PAT"],
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)

completion = client.chat.completions.create(
    model="https://clarifai.com/{user_id}/{app_id}/models/{llm_model_id}",
    messages=[{"role": "user", "content": "Search the web for AI news"}],
    extra_body={"mcp_servers": [
        "https://clarifai.com/clarifai/mcp/models/browser-mcp-server"
    ]},
    stream=True
)
```
