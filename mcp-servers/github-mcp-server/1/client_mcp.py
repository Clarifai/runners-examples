import asyncio
import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="https://api.clarifai.com/v2/ext/mcp/v1/users/{user_id}/apps/{app_id}/models/{model_id}",
    headers={"Authorization": f"Bearer {os.environ['CLARIFAI_PAT']}"},
)

async def main():
    try:
        async with Client(transport) as client:
            tools = await client.list_tools()
            print(f"Available tools: {tools}")
            # TODO: update the dictionary of arguments passed to call_tool to make sense for your MCP.
            result = await client.call_tool("list_pull_requests", {"owner": "clarifai", "repo": "clarifai-python"})
            print(f"Result: {result[0].text}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())