import os
import sys
from typing import Annotated, List, Dict, Optional
from firecrawl import FirecrawlApp
from fastmcp import FastMCP
from pydantic import Field

# Initialize FastMCP server
server = FastMCP(
    "firecrawl-browser-tools-mcp-server",
    instructions="Comprehensive web browsing, scraping, and content extraction tools powered by Firecrawl for intelligent web automation and data gathering",
)


class FirecrawlService:
    def __init__(self):
        firecrawl_api_key = None
        try:
            firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", None)
        except Exception as e:
            raise ValueError(f"User doesn't contain firecrawl_api_key secret: {e}")
        api_key = firecrawl_api_key
        self.app = FirecrawlApp(api_key=api_key)

    def scrape_url(self, url: str, formats: List[str] = ["markdown"], include_raw_html: bool = False, 
                   only_main_content: bool = False, remove_base64_images: bool = False,
                   wait_for: Optional[int] = None, timeout: Optional[int] = None):
        """Scrape a URL with comprehensive options"""
        try:
            # Use the correct Firecrawl API - scrape method with keyword arguments
            result = self.app.scrape(
                url=url,
                formats=formats,
                only_main_content=only_main_content,
                remove_base64_images=remove_base64_images,
                timeout=timeout,
                wait_for=wait_for
            )
            return result
        except Exception as e:
            print(f"Error scraping URL {url}: {e}")
            return None

    def search(self, query: str, limit: int = 10, scrape_options: Optional[Dict] = None,
               country: Optional[str] = None, search_domain: Optional[str] = None):
        """Search the web using Firecrawl with advanced options"""
        try:
            # Use the correct Firecrawl API - search method with keyword arguments
            result = self.app.search(
                query=query,
                limit=limit,
                country=country,
                search_domain=search_domain
            )
            return result
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            return None

    def crawl_url(self, url: str, max_pages: int = 10, include_paths: List[str] = None, 
                  exclude_paths: List[str] = None, allow_external: bool = False,
                  limit: Optional[int] = None, scrape_options: Optional[Dict] = None):
        """Crawl a website with comprehensive options"""
        try:
            # Use the correct Firecrawl API - crawl method with keyword arguments
            result = self.app.crawl(
                url=url,
                limit=max_pages,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                allow_external=allow_external
            )
            return result
        except Exception as e:
            print(f"Error crawling URL {url}: {e}")
            return None

    def map_url(self, url: str, include_paths: List[str] = None, exclude_paths: List[str] = None,
                allow_external: bool = False, limit: Optional[int] = None):
        """Map a website structure without scraping content"""
        try:
            # Use the correct Firecrawl API - map method with keyword arguments
            result = self.app.map(
                url=url,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                allow_external=allow_external,
                limit=limit
            )
            return result
        except Exception as e:
            print(f"Error mapping URL {url}: {e}")
            return None

    def extract_llm(self, url: str, extraction_prompt: str, extraction_schema: Optional[Dict] = None,
                    formats: List[str] = ["markdown"], only_main_content: bool = True):
        """Extract structured data using LLM-powered extraction"""
        try:
            # Use the extract method if available, otherwise fall back to scrape
            if hasattr(self.app, 'extract'):
                result = self.app.extract(
                    url=url,
                    extraction_prompt=extraction_prompt,
                    extraction_schema=extraction_schema
                )
            else:
                # Fall back to scrape with extraction options
                result = self.app.scrape(
                    url=url,
                    formats=formats,
                    only_main_content=only_main_content
                )
            return result
        except Exception as e:
            print(f"Error extracting from URL {url}: {e}")
            return None

    def extract_schema(self, url: str, extraction_schema: Dict, formats: List[str] = ["markdown"],
                       only_main_content: bool = True):
        """Extract structured data using schema-based extraction"""
        try:
            # Use the extract method if available, otherwise fall back to scrape
            if hasattr(self.app, 'extract'):
                result = self.app.extract(
                    url=url,
                    extraction_schema=extraction_schema
                )
            else:
                # Fall back to scrape
                result = self.app.scrape(
                    url=url,
                    formats=formats,
                    only_main_content=only_main_content
                )
            return result
        except Exception as e:
            print(f"Error extracting schema from URL {url}: {e}")
            return None


# Initialize FirecrawlService
try:
    firecrawl_service = FirecrawlService()
    FIRECRAWL_AVAILABLE = True
except Exception as e:
    print(f"Firecrawl not available: {e}", file=sys.stderr)
    firecrawl_service = None
    FIRECRAWL_AVAILABLE = False


@server.tool("scrape_url", description="Scrape content from any URL with comprehensive options")
def scrape_url(
    url: Annotated[str, Field(description="URL to scrape")],
    formats: Annotated[List[str], Field(description="Output formats (markdown, html, text)")] = ["markdown"],
    include_raw_html: Annotated[bool, Field(description="Include raw HTML in response")] = False,
    only_main_content: Annotated[bool, Field(description="Extract only main content, skip navigation/sidebars")] = False,
    remove_base64_images: Annotated[bool, Field(description="Remove base64 encoded images to reduce size")] = False,
    wait_for: Annotated[Optional[int], Field(description="Wait time in milliseconds for page to load")] = None,
    timeout: Annotated[Optional[int], Field(description="Request timeout in milliseconds")] = None,
) -> str:
    """Scrape content from any URL with comprehensive scraping options."""
    if not FIRECRAWL_AVAILABLE:
        return "Error: Firecrawl service not available. Please check FIRECRAWL_API_KEY environment variable."
    
    try:
        result = firecrawl_service.scrape_url(
            url, formats, include_raw_html, only_main_content, 
            remove_base64_images, wait_for, timeout
        )
        
        if not result:
            return f"Failed to scrape URL: {url}"
        
        output = f"Scraped Content from: {url}\n\n"
        
        # Handle the Document object returned by Firecrawl
        title = getattr(result, 'title', 'No title')
        description = getattr(result, 'description', 'No description')
        
        # Get metadata if available
        if hasattr(result, 'metadata') and result.metadata:
            title = getattr(result.metadata, 'title', title)
            description = getattr(result.metadata, 'description', description)
        
        output += f"Title: {title}\n"
        output += f"Description: {description}\n"
        
        # Add content based on requested formats
        if "markdown" in formats and hasattr(result, 'markdown') and result.markdown:
            output += f"\nMarkdown Content:\n{result.markdown}\n"
        
        if "html" in formats and hasattr(result, 'html') and result.html:
            html_preview = result.html[:1000] + "..." if len(result.html) > 1000 else result.html
            output += f"\nHTML Content:\n{html_preview}\n"
        
        if "text" in formats and hasattr(result, 'text') and result.text:
            output += f"\nText Content:\n{result.text}\n"
        
        if include_raw_html and hasattr(result, 'raw_html') and result.raw_html:
            raw_preview = result.raw_html[:1000] + "..." if len(result.raw_html) > 1000 else result.raw_html
            output += f"\nRaw HTML:\n{raw_preview}\n"
        
        if hasattr(result, 'metadata') and result.metadata:
            output += f"\nMetadata: {result.metadata}\n"
        
        return output
        
    except Exception as e:
        return f"Error scraping URL: {str(e)}"


@server.tool("search_web", description="Search the web using Firecrawl's intelligent search capabilities")
def search_web(
    query: Annotated[str, Field(description="Search query")],
    limit: Annotated[int, Field(description="Number of results to return", ge=1, le=20)] = 10,
    formats: Annotated[List[str], Field(description="Output formats for scraped content")] = ["markdown"],
    country: Annotated[Optional[str], Field(description="Country code for localized search (e.g., 'US', 'GB')")] = None,
    search_domain: Annotated[Optional[str], Field(description="Restrict search to specific domain")] = None,
    only_main_content: Annotated[bool, Field(description="Extract only main content from results")] = True,
) -> str:
    """Search the web using Firecrawl's intelligent search capabilities with advanced options."""
    if not FIRECRAWL_AVAILABLE:
        return "Error: Firecrawl service not available. Please check FIRECRAWL_API_KEY environment variable."
    
    try:
        scrape_options = {
            "formats": formats,
            "onlyMainContent": only_main_content
        }
        
        results = firecrawl_service.search(query, limit, scrape_options, country, search_domain)
        
        if not results:
            return f"No results found for search query: '{query}'"
        
        output = f"Web Search Results for '{query}':\n"
        if country:
            output += f"Country: {country}\n"
        if search_domain:
            output += f"Domain: {search_domain}\n"
        output += f"Results: {len(results)}\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"{i}. {result.get('title', 'No title')}\n"
            output += f"   URL: {result.get('url', 'No URL')}\n"
            output += f"   Description: {result.get('description', 'No description')}\n"
            
            if result.get('content'):
                content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                output += f"   Content: {content_preview}\n"
            
            output += "\n"
        
        return output
        
    except Exception as e:
        return f"Error searching web: {str(e)}"


@server.tool("crawl_website", description="Crawl an entire website with comprehensive options")
def crawl_website(
    url: Annotated[str, Field(description="Starting URL for website crawl")],
    max_pages: Annotated[int, Field(description="Maximum number of pages to crawl", ge=1, le=50)] = 10,
    include_paths: Annotated[Optional[List[str]], Field(description="Specific paths to include in crawl")] = None,
    exclude_paths: Annotated[Optional[List[str]], Field(description="Paths to exclude from crawl")] = None,
    allow_external: Annotated[bool, Field(description="Allow crawling external links")] = False,
    formats: Annotated[List[str], Field(description="Output formats for scraped content")] = ["markdown"],
    only_main_content: Annotated[bool, Field(description="Extract only main content")] = True,
) -> str:
    """Crawl an entire website with comprehensive path filtering and content options."""
    if not FIRECRAWL_AVAILABLE:
        return "Error: Firecrawl service not available. Please check FIRECRAWL_API_KEY environment variable."
    
    try:
        scrape_options = {
            "formats": formats,
            "onlyMainContent": only_main_content
        }
        
        results = firecrawl_service.crawl_url(
            url, max_pages, include_paths, exclude_paths, 
            allow_external, None, scrape_options
        )
        
        if not results:
            return f"Failed to crawl website: {url}"
        
        output = f"Website Crawl Results for: {url}\n"
        output += f"Pages crawled: {len(results)}\n"
        if include_paths:
            output += f"Included paths: {', '.join(include_paths)}\n"
        if exclude_paths:
            output += f"Excluded paths: {', '.join(exclude_paths)}\n"
        output += f"Allow external: {allow_external}\n\n"
        
        for i, page in enumerate(results, 1):
            output += f"{i}. {page.get('title', 'No title')}\n"
            output += f"   URL: {page.get('url', 'No URL')}\n"
            output += f"   Content preview: {page.get('content', 'No content')[:200]}...\n\n"
        
        return output
        
    except Exception as e:
        return f"Error crawling website: {str(e)}"


@server.tool("map_website", description="Map a website structure without scraping content")
def map_website(
    url: Annotated[str, Field(description="URL to map website structure")],
    include_paths: Annotated[Optional[List[str]], Field(description="Specific paths to include in mapping")] = None,
    exclude_paths: Annotated[Optional[List[str]], Field(description="Paths to exclude from mapping")] = None,
    allow_external: Annotated[bool, Field(description="Allow mapping external links")] = False,
    limit: Annotated[Optional[int], Field(description="Maximum number of pages to map", ge=1, le=100)] = None,
) -> str:
    """Map a website structure to understand its organization without scraping content."""
    if not FIRECRAWL_AVAILABLE:
        return "Error: Firecrawl service not available. Please check FIRECRAWL_API_KEY environment variable."
    
    try:
        results = firecrawl_service.map_url(url, include_paths, exclude_paths, allow_external, limit)
        
        if not results:
            return f"Failed to map website: {url}"
        
        output = f"Website Map for: {url}\n"
        output += f"Pages mapped: {len(results)}\n"
        if include_paths:
            output += f"Included paths: {', '.join(include_paths)}\n"
        if exclude_paths:
            output += f"Excluded paths: {', '.join(exclude_paths)}\n"
        output += f"Allow external: {allow_external}\n\n"
        
        for i, page in enumerate(results, 1):
            output += f"{i}. {page.get('title', 'No title')}\n"
            output += f"   URL: {page.get('url', 'No URL')}\n"
            if page.get('links'):
                output += f"   Links found: {len(page['links'])}\n"
            output += "\n"
        
        return output
        
    except Exception as e:
        return f"Error mapping website: {str(e)}"


@server.tool("extract_with_llm", description="Extract structured data using LLM-powered extraction")
def extract_with_llm(
    url: Annotated[str, Field(description="URL to extract data from")],
    extraction_prompt: Annotated[str, Field(description="Custom prompt for data extraction")],
    extraction_schema: Annotated[Optional[Dict], Field(description="JSON schema for structured extraction")] = None,
    formats: Annotated[List[str], Field(description="Output formats")] = ["markdown"],
    only_main_content: Annotated[bool, Field(description="Extract only main content")] = True,
) -> str:
    """Extract structured data using LLM-powered extraction with custom prompts."""
    if not FIRECRAWL_AVAILABLE:
        return "Error: Firecrawl service not available. Please check FIRECRAWL_API_KEY environment variable."
    
    try:
        result = firecrawl_service.extract_llm(
            url, extraction_prompt, extraction_schema, formats, only_main_content
        )
        
        if not result:
            return f"Failed to extract data from: {url}"
        
        output = f"LLM Extraction Results from: {url}\n"
        output += f"Extraction Prompt: {extraction_prompt}\n"
        if extraction_schema:
            output += f"Schema: {extraction_schema}\n"
        output += "\n"
        
        output += f"Title: {result.get('title', 'No title')}\n"
        output += f"Description: {result.get('description', 'No description')}\n"
        
        if result.get('llm_extraction'):
            output += f"Extracted Data:\n{result['llm_extraction']}\n"
        else:
            output += f"Content:\n{result.get('content', 'No content')}\n"
        
        if result.get('metadata'):
            output += f"\nMetadata: {result['metadata']}\n"
        
        return output
        
    except Exception as e:
        return f"Error extracting with LLM: {str(e)}"


@server.tool("extract_with_schema", description="Extract structured data using schema-based extraction")
def extract_with_schema(
    url: Annotated[str, Field(description="URL to extract data from")],
    extraction_schema: Annotated[Dict, Field(description="JSON schema defining the data structure to extract")],
    formats: Annotated[List[str], Field(description="Output formats")] = ["markdown"],
    only_main_content: Annotated[bool, Field(description="Extract only main content")] = True,
) -> str:
    """Extract structured data using predefined JSON schema for consistent data extraction."""
    if not FIRECRAWL_AVAILABLE:
        return "Error: Firecrawl service not available. Please check FIRECRAWL_API_KEY environment variable."
    
    try:
        result = firecrawl_service.extract_schema(
            url, extraction_schema, formats, only_main_content
        )
        
        if not result:
            return f"Failed to extract schema data from: {url}"
        
        output = f"Schema Extraction Results from: {url}\n"
        output += f"Schema: {extraction_schema}\n\n"
        
        output += f"Title: {result.get('title', 'No title')}\n"
        output += f"Description: {result.get('description', 'No description')}\n"
        
        if result.get('schema_extraction'):
            output += f"Extracted Data:\n{result['schema_extraction']}\n"
        else:
            output += f"Content:\n{result.get('content', 'No content')}\n"
        
        if result.get('metadata'):
            output += f"\nMetadata: {result['metadata']}\n"
        
        return output
        
    except Exception as e:
        return f"Error extracting with schema: {str(e)}"


# Static resource for Firecrawl configuration
@server.resource("config://firecrawl_settings")
def get_firecrawl_settings():
    return {
        "service_available": FIRECRAWL_AVAILABLE,
        "supported_formats": ["markdown", "html", "text"],
        "max_search_results": 20,
        "max_crawl_pages": 50,
        "max_map_pages": 100,
        "default_timeout": 30,
        "extraction_modes": ["llm-extraction", "schema-extraction"],
        "available_tools": [
            "scrape_url",
            "search_web",
            "crawl_website", 
            "map_website",
            "extract_with_llm",
            "extract_with_schema"
        ],
        "features": [
            "intelligent_content_extraction",
            "javascript_rendering",
            "structured_data_extraction",
            "website_mapping",
            "advanced_search",
            "path_filtering",
            "external_link_handling"
        ]
    }


# Dynamic resource for website information
@server.resource("site://{domain}/browser_info")
def get_site_browser_info(domain: str):
    return {
        "domain": domain,
        "note": "Use Firecrawl browser tools to interact with this website",
        "available_operations": [
            "scrape_url - Extract content from specific pages",
            "search_web - Search for content across the web",
            "crawl_website - Crawl entire website structure",
            "map_website - Map website without content extraction",
            "extract_with_llm - Extract data using AI prompts",
            "extract_with_schema - Extract data using JSON schemas"
        ],
        "recommended_workflow": [
            "1. Use map_website to understand site structure",
            "2. Use scrape_url for specific page content",
            "3. Use extract_with_llm for custom data extraction",
            "4. Use crawl_website for comprehensive analysis"
        ]
    }


@server.prompt()
def web_browsing_prompt(task_type: str) -> str:
    """Generate prompts for web browsing and data extraction tasks."""
    prompts = {
        "content_research": "To research web content:\n1. Use search_web to find relevant pages\n2. Use scrape_url for detailed content extraction\n3. Use extract_with_llm for specific information extraction\n4. Use crawl_website for comprehensive site analysis",
        "data_extraction": "To extract structured data:\n1. Use extract_with_llm for custom prompt-based extraction\n2. Use extract_with_schema for consistent schema-based extraction\n3. Use scrape_url with specific formats for raw content\n4. Use map_website to understand site structure first",
        "website_analysis": "To analyze a website:\n1. Use map_website to understand site structure\n2. Use crawl_website for comprehensive content analysis\n3. Use scrape_url for specific page deep-dives\n4. Use extract_with_llm for insights and summaries",
        "competitive_research": "To research competitors:\n1. Use search_web to find competitor websites\n2. Use map_website to understand their site structure\n3. Use crawl_website to analyze their content\n4. Use extract_with_llm to compare features/pricing",
        "market_research": "To conduct market research:\n1. Use search_web for industry trends and news\n2. Use crawl_website to analyze industry websites\n3. Use extract_with_llm for market insights\n4. Use extract_with_schema for consistent data collection",
        "content_monitoring": "To monitor web content:\n1. Use scrape_url for regular content checks\n2. Use crawl_website for comprehensive site monitoring\n3. Use extract_with_llm for change detection\n4. Use search_web for new content discovery"
    }
    
    return prompts.get(task_type, f"Web browsing guidance for: {task_type}")


# MCP Model Class for Clarifai integration
from clarifai.runners.models.mcp_class import MCPModelClass


class FirecrawlModelClass(MCPModelClass):
    def get_server(self) -> FastMCP:
        return server


# Main function to run the MCP server
if __name__ == "__main__":
    import asyncio
    import sys
    
    # Simple approach - just run the server
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("Server stopped by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
