import asyncio
import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="https://api.clarifai.com/v2/ext/mcp/v1/users/luv_2261/apps/mcp/models/firecrawl-browser-tools-mcp-server",
    headers={"Authorization": "Bearer " + os.environ["CLARIFAI_PAT"]},
)

async def main():
    async with Client(transport) as client:
        tools = await client.list_tools()
        # print(f"Available tools: {tools}")
        # TODO: update the dictionary of arguments passed to call_tool to make sense for your MCP.
        result = await client.call_tool(tools[0].name, arguments={"url": "https://www.clarifai.com/"})
        print(f"Result: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())