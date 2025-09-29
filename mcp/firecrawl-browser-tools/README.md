# Firecrawl Browser Tools MCP Server

A comprehensive MCP (Model Control Protocol) server that provides advanced web browsing, scraping, search, and content extraction capabilities powered by [Firecrawl](https://firecrawl.dev).

## Configuration

Must have `FIRECRAWL_API_KEY` in your secrets! and use `secrets` section in `config.yaml`

```
secrets:
  - id: "firecrawl_api_key"
    type: "env"
    env_var: "FIRECRAWL_API_KEY"
    description: "API key of Firecrawl for scraping and browsing"
```


### Deployment to Clarifai

The server includes a `config.yaml` file for deployment to Clarifai's model hosting platform:

```bash
clarifai model upload
```


## Features

### 🔍 **Intelligent Web Search**
- Advanced web search with content scraping
- Country-specific and domain-restricted search
- Intelligent content filtering and extraction
- Support for complex search queries

### 🌐 **Advanced Web Scraping**
- Scrape any URL with multiple output formats (markdown, HTML, text)
- Intelligent content extraction and cleaning
- Support for JavaScript-heavy websites
- Configurable content filtering (main content only, base64 image removal)
- Custom wait times and timeouts

### 🗺️ **Website Mapping**
- Map website structure without content extraction
- Path-based inclusion and exclusion filtering
- External link handling options
- Site organization analysis

### 🕷️ **Website Crawling**
- Crawl entire websites with comprehensive options
- Path-based filtering (include/exclude specific paths)
- External link crawling control
- Configurable page limits and content formats
- Full site analysis and content extraction

### 🤖 **LLM-Powered Data Extraction**
- Custom prompt-based data extraction
- Intelligent content understanding
- Flexible extraction schemas
- Natural language extraction instructions

### 📋 **Schema-Based Data Extraction**
- JSON schema-driven data extraction
- Consistent structured data output
- Type-safe extraction results
- Automated data validation

### Available Tools

#### 1. `scrape_url`
Scrape content from any URL with comprehensive options.

```python
# Example usage
result = await session.call_tool(
    "scrape_url",
    arguments={
        "url": "https://example.com",
        "formats": ["markdown", "html"],
        "include_raw_html": False,
        "only_main_content": True,
        "remove_base64_images": True,
        "wait_for": 2000,
        "timeout": 30000
    }
)
```

#### 2. `search_web`
Search the web using Firecrawl's intelligent search capabilities.

```python
# Example usage
result = await session.call_tool(
    "search_web",
    arguments={
        "query": "Python web scraping tutorial",
        "limit": 10,
        "formats": ["markdown"],
        "country": "US",
        "search_domain": "github.com",
        "only_main_content": True
    }
)
```

#### 3. `crawl_website`
Crawl an entire website with comprehensive options.

```python
# Example usage
result = await session.call_tool(
    "crawl_website",
    arguments={
        "url": "https://example.com",
        "max_pages": 10,
        "include_paths": ["/products", "/pricing"],
        "exclude_paths": ["/admin", "/private"],
        "allow_external": False,
        "formats": ["markdown"],
        "only_main_content": True
    }
)
```

#### 4. `map_website`
Map a website structure without scraping content.

```python
# Example usage
result = await session.call_tool(
    "map_website",
    arguments={
        "url": "https://example.com",
        "include_paths": ["/docs", "/api"],
        "exclude_paths": ["/admin"],
        "allow_external": False,
        "limit": 20
    }
)
```

#### 5. `extract_with_llm`
Extract structured data using LLM-powered extraction.

```python
# Example usage
result = await session.call_tool(
    "extract_with_llm",
    arguments={
        "url": "https://example.com",
        "extraction_prompt": "Extract the main heading, description, and contact information",
        "extraction_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "contact": {"type": "string"}
            }
        },
        "formats": ["markdown"],
        "only_main_content": True
    }
)
```

#### 6. `extract_with_schema`
Extract structured data using predefined JSON schema.

```python
# Example usage
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "price": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}}
    }
}

result = await session.call_tool(
    "extract_with_schema",
    arguments={
        "url": "https://example.com",
        "extraction_schema": schema,
        "formats": ["markdown"],
        "only_main_content": True
    }
)
```

### Resources

#### `config://firecrawl_settings`
Get Firecrawl configuration and capabilities.

#### `site://{domain}/browser_info`
Get information about available browser tools for a specific domain.

### Prompts

The server provides research prompts for different use cases:

- `content_research`: Web content research and analysis
- `data_extraction`: Structured data extraction workflows
- `website_analysis`: Comprehensive website analysis
- `competitive_research`: Competitive intelligence gathering
- `market_research`: Market analysis and trend research
- `content_monitoring`: Web content monitoring and change detection


## Support

- **Firecrawl Documentation**: [docs.firecrawl.dev](https://docs.firecrawl.dev)
- **MCP Protocol**: [Model Control Protocol](https://modelcontextprotocol.io)
- **Issues**: Report issues in the repository
