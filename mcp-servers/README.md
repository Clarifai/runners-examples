# Model Context Protocol (MCP) Server Examples

This directory contains examples for deploying MCP servers on Clarifai. MCP servers provide tool-calling capabilities that enable AI agents to interact with external systems, execute code, browse the web, and more.

## What is MCP?

The Model Context Protocol (MCP) is an open standard for connecting AI models to external tools and data sources. MCP servers:
- Expose tools that AI agents can call
- Handle authentication and authorization
- Integrate with external APIs and services
- Enable agentic workflows

## Available Examples

This directory includes 11 MCP server examples:

**Simple Tools:**
- **browser-mcp-server** - Web browsing with DuckDuckGo search
- **math** - Mathematical calculations
- **web-search** - Web search capabilities

**API Integration:**
- **github-mcp-server** - GitHub API integration
- **postgres** - PostgreSQL database access
- **google-drive** - Google Drive file operations
- **slack-tools-server** - Slack messaging integration

**Advanced Features:**
- **code-execution-docker-version** - Sandboxed code execution with Docker
- **code-execution-without-docker-version** - Direct code execution
- **browser-tools** - Advanced browser automation
- **firecrawl-browser-tools** - Web scraping with Firecrawl

## Feature Matrix

See [INDEX.md](INDEX.md) for a detailed feature comparison of all MCP servers.

## Quick Reference

| Feature | Examples |
|---------|----------|
| **Browser Automation** | browser-mcp-server, browser-tools, firecrawl-browser-tools |
| **API Integration** | github-mcp-server, slack-tools-server, google-drive |
| **Code Execution** | code-execution-docker-version, code-execution-without-docker-version |
| **Database Access** | postgres |
| **Search** | browser-mcp-server, web-search |

## Choosing the Right MCP Server

**If you want to...**

- **Browse the web** → `browser-mcp-server`
- **Perform calculations** → `math`
- **Search the web** → `web-search`
- **Integrate with GitHub** → `github-mcp-server`
- **Query databases** → `postgres`
- **Access Google Drive** → `google-drive`
- **Send Slack messages** → `slack-tools-server`
- **Execute code safely** → `code-execution-docker-version`
- **Scrape websites** → `firecrawl-browser-tools`
- **Automate browsers** → `browser-tools`

## MCP Server Architecture

```
┌─────────────────┐
│  Agentic Model  │  (AgenticModelClass)
└────────┬────────┘
         │ calls tools via MCP
         ▼
┌─────────────────┐
│   MCP Server    │  (StdioMCPModelClass)
└────────┬────────┘
         │ executes
         ▼
┌─────────────────┐
│ External System │  (API, Database, Browser, etc.)
└─────────────────┘
```

## Configuration

### Model Class
All MCP servers use `StdioMCPModelClass`:

```yaml
model:
  id: your-mcp-server
  model_type_id: studio-text-to-text
  user_id: your-user-id
  app_id: your-app-id
  model_version:
    id: version-1
    inference_compute_info:
      cpu_limit: "2"
      cpu_memory: "4Gi"
```

### Environment Variables
MCP servers often require API keys or credentials:

```yaml
inference_compute_info:
  env_vars:
    - name: GITHUB_TOKEN
      value: ghp_xxxxx
    - name: DATABASE_URL
      value: postgresql://user:pass@host/db
```

### No GPU Required
MCP servers typically don't need GPU resources - they orchestrate tools and APIs rather than running ML models.

## Deployment Workflow

1. **Choose an MCP server** based on required capabilities
2. **Set up credentials** for external services (API keys, tokens)
3. **Configure environment variables** in config.yaml
4. **Test locally**: `clarifai model test`
5. **Deploy**: `clarifai model upload`
6. **Integrate with agentic model** - Connect to AgenticModelClass

## Using MCP Servers with Agentic Models

MCP servers are designed to work with agentic models:

```python
from clarifai.client.model import Model

# Your agentic model
agentic_model = Model(
    user_id="your-user-id",
    app_id="your-app-id",
    model_id="your-agentic-model"
)

# Configure MCP servers as tools
response = agentic_model.predict(
    inputs=[{
        "data": {
            "text": {"raw": "Search GitHub for Python MCP examples"}
        }
    }],
    inference_params={
        "mcp_servers": ["github-mcp-server"]
    }
)
```

## Common Patterns

### Tool Definition

```python
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description for the AI agent."""
    # Tool implementation
    return result
```

### Error Handling

```python
@mcp.tool()
def safe_tool(input: str) -> dict:
    """Tool with error handling."""
    try:
        result = perform_operation(input)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Authentication

```python
import os

@mcp.tool()
def authenticated_tool(query: str) -> str:
    """Tool requiring authentication."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        return "Error: API_KEY not configured"

    # Use API key...
    return result
```

## Security Considerations

### 01-basic Servers
- Minimal security concerns
- No credentials required
- Safe for public deployment

### 02-integration Servers
- **Requires API keys** - Store securely in environment variables
- **Rate limiting** - Implement to prevent abuse
- **Scope permissions** - Use minimal required permissions

### 03-advanced Servers
- **Code execution risks** - Use Docker for sandboxing
- **Resource limits** - Set CPU/memory limits
- **Input validation** - Sanitize all inputs
- **Timeout controls** - Prevent infinite loops

## Performance Tips

- **Connection pooling**: Reuse API connections where possible
- **Caching**: Cache frequently accessed data
- **Async operations**: Use async/await for I/O operations
- **Timeouts**: Set reasonable timeouts for external calls
- **Error recovery**: Implement retry logic for transient failures

## Common Use Cases

### Developer Tools
- **GitHub integration**: Create issues, review PRs, search code
- **Code execution**: Test code snippets, run scripts
- **Database queries**: Analyze data, run reports

### Productivity
- **Slack integration**: Send notifications, create channels
- **Google Drive**: Access documents, create files
- **Web search**: Research topics, gather information

### Automation
- **Browser automation**: Fill forms, scrape data
- **Workflow orchestration**: Chain multiple tools
- **Data processing**: Transform and analyze data

## Development Workflow

### Local Testing

1. Install dependencies:
   ```bash
   cd mcp-servers/01-basic/math
   pip install -r requirements.txt
   ```

2. Test the MCP server:
   ```bash
   python 1/model.py
   ```

3. Test with Clarifai CLI:
   ```bash
   clarifai model test --input '{"text": "calculate 2+2"}'
   ```

### Debugging

- Check logs for tool execution traces
- Validate tool definitions are correctly exposed
- Test external API connections separately
- Use print statements in tool implementations

## Additional Resources

- [Getting Started](../00-getting-started/) - Learn deployment basics
- [LLM Examples](../llm/) - Agentic models that use MCP servers
- [FastMCP Documentation](https://github.com/jlowin/fastmcp) - Framework docs
- [MCP Specification](https://modelcontextprotocol.io/) - Protocol details

## Contributing

When adding MCP server examples:
1. Place in appropriate complexity directory
2. Include README with frontmatter metadata
3. Document required environment variables
4. Provide example tool usage
5. Include security considerations
6. Add to feature matrix in INDEX.md
