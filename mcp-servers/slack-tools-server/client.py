# import asyncio
# import os
# import tempfile

# from clarifai.urls.helper import ClarifaiUrlHelper
# from fastmcp import Client
# from fastmcp.client.transports import StreamableHttpTransport

# PAT = os.environ['CLARIFAI_PAT']
# url = ClarifaiUrlHelper().mcp_api_url()  # get url from the current clarifai config

# transport = StreamableHttpTransport(url=url, headers={"Authorization": "Bearer " + PAT})


# async def run_slack_examples():
#     """Run comprehensive Slack MCP server examples."""
#     print("🚀 Slack MCP Server Examples")
#     print("=" * 60)

#     async with Client(transport) as client:
#         # List available tools first
#         print("📋 Available Tools:")
#         tools = await client.list_tools()
#         for tool in tools:
#             print(f"   • {tool.name}: {tool.description}")
#         print("\n" + "=" * 60 + "\n")

#         # Example 1: Get workspace information
#         print("🏢 Example 1: Getting workspace information")
#         try:
#             result = await client.call_tool("slack_get_workspace_info", {})
#             print("✅ Workspace Info:")
#             print(result.content[0].text)
#         except Exception as e:
#             print(f"❌ Error: {e}")
#         print("\n" + "=" * 60 + "\n")

#         # Example 2: List channels
#         print("📺 Example 2: Listing channels")
#         try:
#             result = await client.call_tool("slack_list_channels", {"limit": 5})
#             print("✅ Channels:")
#             print(result.content[0].text)
#         except Exception as e:
#             print(f"❌ Error: {e}")
#         print("\n" + "=" * 60 + "\n")

#         # Example 3: List users
#         print("👥 Example 3: Listing users")
#         try:
#             result = await client.call_tool("slack_list_users", {"limit": 5})
#             print("✅ Users:")
#             print(result.content[0].text)
#         except Exception as e:
#             print(f"❌ Error: {e}")
#         print("\n" + "=" * 60 + "\n")

#         # # Example 4: Send a message
#         # print("💬 Example 4: Sending a message")
#         # try:
#         #     result = await client.call_tool("slack_send_message", {
#         #         "channel": "general",
#         #         "text": "Hello from the Slack MCP server! 🚀"
#         #     })
#         #     print("✅ Message Sent:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 5: Get channel information
#         # print("ℹ️  Example 5: Getting channel information")
#         # try:
#         #     result = await client.call_tool("slack_get_channel_info", {
#         #         "channel": "general",
#         #         "include_num_members": True
#         #     })
#         #     print("✅ Channel Info:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 6: List messages from a channel
#         # print("📝 Example 6: Listing recent messages")
#         # try:
#         #     result = await client.call_tool("slack_list_messages", {
#         #         "channel": "general",
#         #         "limit": 5
#         #     })
#         #     print("✅ Recent Messages:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 7: Create a channel
#         # print("📂 Example 7: Creating a channel")
#         # try:
#         #     result = await client.call_tool("slack_create_channel", {
#         #         "name": "mcp-test-channel",
#         #         "is_private": False,
#         #         "topic": "Testing MCP Slack integration",
#         #         "purpose": "Channel for testing MCP Slack server functionality"
#         #     })
#         #     print("✅ Channel Creation:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 8: Upload a file
#         # print("📤 Example 8: Uploading a file")
        
#         # # Create a temporary file for demonstration
#         # with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
#         #     f.write("This is a test file for Slack upload demonstration.\nCreated by MCP Slack server.")
#         #     temp_file_path = f.name

#         # try:
#         #     result = await client.call_tool("slack_upload_file", {
#         #         "channel": "general",
#         #         "file_path": temp_file_path,
#         #         "title": "MCP Test File",
#         #         "initial_comment": "Test file uploaded by MCP Slack server"
#         #     })
#         #     print("✅ File Upload:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # finally:
#         #     # Clean up temporary file
#         #     os.unlink(temp_file_path)
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 9: List files
#         # print("📁 Example 9: Listing files")
#         # try:
#         #     result = await client.call_tool("slack_list_files", {
#         #         "count": 5,
#         #         "types": "all"
#         #     })
#         #     print("✅ Files:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 10: Set channel topic
#         # print("📌 Example 10: Setting channel topic")
#         # try:
#         #     result = await client.call_tool("slack_set_channel_topic", {
#         #         "channel": "general",
#         #         "topic": "General discussion - Updated by MCP server"
#         #     })
#         #     print("✅ Topic Update:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 11: Set channel purpose
#         # print("🎯 Example 11: Setting channel purpose")
#         # try:
#         #     result = await client.call_tool("slack_set_channel_purpose", {
#         #         "channel": "general",
#         #         "purpose": "General workspace communication and announcements"
#         #     })
#         #     print("✅ Purpose Update:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 12: Get user information
#         # print("👤 Example 12: Getting user information")
#         # try:
#         #     result = await client.call_tool("slack_get_user_info", {
#         #         "user": "U1234567890",  # Mock user ID
#         #         "include_locale": True
#         #     })
#         #     print("✅ User Info:")
#         #     print(result.content[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 13: Available resources
#         # print("📚 Example 13: Available resources")
#         # try:
#         #     resources = await client.list_resources()
#         #     print("✅ Available Resources:")
#         #     for resource in resources:
#         #         print(f"   • {resource.uri}: {resource.name}")
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 14: Slack settings
#         # print("⚙️  Example 14: Slack configuration")
#         # try:
#         #     result = await client.read_resource("config://slack_settings")
#         #     print("✅ Slack Settings:")
#         #     print(result.contents[0].text)
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")
#         # print("\n" + "=" * 60 + "\n")

#         # # Example 15: Available prompts
#         # print("💡 Example 15: Available prompts")
#         # try:
#         #     prompts = await client.list_prompts()
#         #     print("✅ Available Prompts:")
#         #     for prompt in prompts:
#         #         print(f"   • {prompt.name}: {prompt.description}")
#         # except Exception as e:
#         #     print(f"❌ Error: {e}")

#     print("\n" + "=" * 60)
#     print("🎉 All examples completed!")
#     print("\n💡 Tips:")
#     print("   • Set SLACK_BOT_TOKEN for full functionality")
#     print("   • Use slack_send_message for team communication")
#     print("   • Use slack_create_channel for project organization")
#     print("   • Use slack_upload_file for document sharing")
#     print("   • Use slack_list_messages to monitor conversations")
#     print("   • Use slack_invite_to_channel for team management")
#     print("   • This example uses mock data for demonstration")
#     print("=" * 60)


# if __name__ == "__main__":
#     asyncio.run(run_slack_examples())


import asyncio
import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="https://api.clarifai.com/v2/ext/mcp/v1/users/luv_2261/apps/mcp/models/slack-mcp-server",
    headers={"Authorization": "Bearer " + os.environ["CLARIFAI_PAT"]},
)

async def main():
    async with Client(transport) as client:
        tools = await client.list_tools()
        # print(f"Available tools: {tools}")
        # TODO: update the dictionary of arguments passed to call_tool to make sense for your MCP.
        result = await client.call_tool("slack_list_messages", {
                "channel": "general",
                # "text": "Hello from the Slack MCP server! 🚀"
            })
        print(f"Result: {result.content[0].text}")
        

if __name__ == "__main__":
    asyncio.run(main())