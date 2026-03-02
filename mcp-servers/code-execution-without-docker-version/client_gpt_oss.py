

import asyncio
import json
import os
from typing import Any, Dict
from clarifai.urls.helper import ClarifaiUrlHelper
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from openai import AsyncOpenAI

# Configuration
os.environ["CLARIFAI_PAT"] = "YOUR_PAT_HERE"
MODEL_NAME = "https://clarifai.com/openai/chat-completion/models/gpt-oss-120b"
MCP_SERVER_URL = "YOUR_MCP_SERVER_URL_HERE"


# Initialize clients
openai_client = AsyncOpenAI(api_key=os.environ["CLARIFAI_PAT"], base_url="https://api.clarifai.com/v2/ext/openai/v1")
mcp_client = None
mcp_tools = []

async def connect_to_mcp():
    """Connect to the Python code execution MCP server."""
    global mcp_client, mcp_tools

    transport = StreamableHttpTransport(
        url=MCP_SERVER_URL,
        headers={"Authorization": "Bearer " + os.environ["CLARIFAI_PAT"]}
    )

    mcp_client = Client(transport)
    await mcp_client.__aenter__()

    # Get available tools
    tools_result = await mcp_client.list_tools()
    mcp_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools_result
    ]

    return mcp_tools

async def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute an MCP tool call."""
    try:
        result = await mcp_client.call_tool(tool_name, arguments)
        return result.content[0].text
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

async def test_queries():
    """Test with two simple queries."""

    test_queries = [
        "Use time package in python to print the current time",
        "Use the numpy python package to perform a matrix multiplication of [[1, 2], [3, 4]] and [[5, 6], [7, 8]] and print the result"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Query {i}: {query}")
        print('='*50)

        try:
            response = await openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Python assistant with code execution tools."
                    },
                    {"role": "user", "content": query}
                ],
                tools=mcp_tools,
                n=2,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=500
            )

            message = response.choices[0].message
            print(f"Assistant: {message.content}")

            if message.tool_calls:
                print(f"\n🔧 Tool calls: {len(message.tool_calls)}")

                tool_responses = []
                for tool_call in message.tool_calls:
                    print(f"\n--- {tool_call.function.name} ---")
                    args = json.loads(tool_call.function.arguments)

                    if "code" in args:
                        print(f"Code: {args['code']}")
                    if "packages" in args:
                        print(f"Packages: {args['packages']}")

                    result = await execute_mcp_tool(tool_call.function.name, args)
                    tool_responses.append(result)
                    print(f"Result:\n{result}")

                # Get final response
                follow_up_messages = [
                    {"role": "system", "content": "You are a Python assistant with code execution tools."},
                    {"role": "user", "content": query},
                    message
                ]

                for tool_call, tool_result in zip(message.tool_calls, tool_responses):
                    follow_up_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                final_response = await openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=follow_up_messages,
                    temperature=0.1,
                    max_tokens=300
                )

                print(f"\n✅ Final: {final_response.choices[0].message.content}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def cleanup():
    """Clean up MCP connection."""
    global mcp_client
    if mcp_client:
        await mcp_client.__aexit__(None, None, None)
        mcp_client = None

async def main():
    """Main function"""
    print("🚀 Testing GPT-OSS with Python Execution Tools")

    try:
        await connect_to_mcp()
        print(f"✅ Connected! Tools: {len(mcp_tools)}")
        await test_queries()
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())