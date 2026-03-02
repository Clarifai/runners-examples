import asyncio
import json
import os
from typing import Any, Dict, List

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
    """Connect to the math MCP server."""
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

async def test_math_tools():
    """Test math tools with simple queries."""

    test_queries = [
        "What is 15.5 + 23.2?",
        "Calculate 100 divided by 8"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")

        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to math tools. Use the appropriate tools when needed."
                },
                {"role": "user", "content": query}
            ],
            tools=mcp_tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=500
        )

        message = response.choices[0].message

        if message.tool_calls:
            print(f"Tool calls: {len(message.tool_calls)}")

            tool_responses = []
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = await execute_mcp_tool(tool_call.function.name, args)
                tool_responses.append(result)
                print(f"Tool {tool_call.function.name}: {result}")

            # Get final response
            follow_up_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to math tools. Use the appropriate tools when needed."
                },
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
                max_tokens=500
            )

            print(f"Final answer: {final_response.choices[0].message.content}")
        else:
            print(f"Direct answer: {message.content}")


async def cleanup():
    """Clean up MCP connection."""
    global mcp_client
    if mcp_client:
        await mcp_client.__aexit__(None, None, None)
        mcp_client = None

async def main():
    """Main function"""
    print("Testing GPT-OSS with Math MCP Tools")

    try:
        # Connect to MCP server
        await connect_to_mcp()
        print(f"Connected to math MCP server with {len(mcp_tools)} tools")

        # Test the tools
        await test_math_tools()

    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())