import os
import sys
from typing import Annotated, List, Dict, Optional, Union
from clarifai.utils.logging import logger
from fastmcp import FastMCP
from pydantic import Field

# Initialize FastMCP server
server = FastMCP(
    "slack-mcp-server",
    instructions="Comprehensive Slack operations for messaging, channels, users, files, workflows, and advanced team collaboration with full API integration",
)

slack_token = os.getenv("SLACK_BOT_TOKEN")
class SlackService:
    """Comprehensive Slack service with Clarifai integration."""
    
    def __init__(self):
        """Initialize Slack service with Clarifai credentials."""
        try:
            
            # Try to get Slack credentials from environment or Clarifai user secrets
            slack_token = os.getenv("SLACK_BOT_TOKEN")
            if not slack_token:
                logger.warning("No Slack token found, using mock data")
                self.client = None
                self.available = False
                return
            
            self.client = self._create_slack_client(slack_token)
            self.available = self.client is not None
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack service: {e}")
            self.client = None
            self.available = False
    
    def _create_slack_client(self, token):
        """Create and return a Slack client object."""
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
            
            client = WebClient(token=token)
            
            # Test the connection
            response = client.auth_test()
            logger.info(f"Connected to Slack workspace: {response.get('team', 'Unknown')}")
            
            return client
            
        except ImportError:
            logger.error("Slack SDK not available. Install with: pip install slack-sdk")
            return None
        except Exception as e:
            logger.error(f"Failed to create Slack client: {str(e)}")
            return None

# Initialize Slack service
try:
    slack_service = SlackService()
    SLACK_AVAILABLE = slack_service.available
except Exception as e:
    logger.error(f"Slack service initialization failed: {e}")
    slack_service = None
    SLACK_AVAILABLE = False


def handle_slack_operation(operation_func):
    """Decorator to handle Slack operations with proper error handling."""
    def wrapper(*args, **kwargs):
        try:
            if not SLACK_AVAILABLE or not slack_service or not slack_service.client:
                # Return mock data if service not available
                return operation_func(None, *args, **kwargs)
            
            return operation_func(slack_service.client, *args, **kwargs)
        except Exception as e:
            return f"Slack operation failed: {str(e)}"
    return wrapper


@server.tool("slack_send_message", description="Send a message to a Slack channel or user")
def slack_send_message(
    channel: Annotated[str, Field(description="Channel ID, channel name, or user ID to send message to")],
    text: Annotated[str, Field(description="Message text to send")],
    thread_ts: Annotated[str, Field(description="Timestamp of parent message for thread reply")] = "",
    blocks: Annotated[str, Field(description="JSON string of Slack blocks for rich formatting")] = "",
    attachments: Annotated[str, Field(description="JSON string of attachments")] = "",
) -> str:
    """Send a message to a Slack channel or user."""
    
    @handle_slack_operation
    def _send_message(client, channel, text, thread_ts, blocks, attachments):
        if not client:
            return f"Mock message sent to {channel}:\n{text}\n\nNote: Install slack-sdk and set SLACK_BOT_TOKEN for real messaging."
        
        try:
            message_kwargs = {
                "channel": channel,
                "text": text
            }
            
            if thread_ts:
                message_kwargs["thread_ts"] = thread_ts
            
            if blocks:
                import json
                message_kwargs["blocks"] = json.loads(blocks)
            
            if attachments:
                import json
                message_kwargs["attachments"] = json.loads(attachments)
            
            response = client.chat_postMessage(**message_kwargs)
            
            return (
                f"✅ Message sent successfully!\n"
                f"Channel: {channel}\n"
                f"Message: {text}\n"
                f"Timestamp: {response['ts']}\n"
                f"Message URL: {response.get('permalink', 'N/A')}"
            )
            
        except Exception as e:
            return f"Error sending message: {str(e)}"
    
    return _send_message(channel, text, thread_ts, blocks, attachments)


@server.tool("slack_list_channels", description="List all channels in the Slack workspace")
def slack_list_channels(
    types: Annotated[str, Field(description="Channel types: public, private, mpim, im")] = "public_channel,private_channel",
    exclude_archived: Annotated[bool, Field(description="Exclude archived channels")] = True,
    limit: Annotated[int, Field(description="Maximum number of channels to return", ge=1, le=1000)] = 100,
) -> str:
    """List all channels in the Slack workspace."""
    
    @handle_slack_operation
    def _list_channels(client, types, exclude_archived, limit):
        if not client:
            mock_channels = [
                {"id": "C1234567890", "name": "general", "is_channel": True, "is_private": False, "num_members": 25},
                {"id": "C2345678901", "name": "random", "is_channel": True, "is_private": False, "num_members": 15},
                {"id": "C3456789012", "name": "dev-team", "is_channel": True, "is_private": True, "num_members": 8},
            ]
            
            output = f"Slack Channels (showing {len(mock_channels)} channels):\n\n"
            for channel in mock_channels:
                output += f"• #{channel['name']}\n"
                output += f"  ID: {channel['id']}\n"
                output += f"  Type: {'Private' if channel['is_private'] else 'Public'}\n"
                output += f"  Members: {channel['num_members']}\n\n"
            
            return output
        
        try:
            response = client.conversations_list(
                types=types,
                exclude_archived=exclude_archived,
                limit=limit
            )
            
            channels = response.get('channels', [])
            
            if not channels:
                return "No channels found in the workspace"
            
            output = f"Slack Channels ({len(channels)} channels):\n\n"
            for channel in channels:
                output += f"• #{channel['name']}\n"
                output += f"  ID: {channel['id']}\n"
                output += f"  Type: {'Private' if channel.get('is_private', False) else 'Public'}\n"
                output += f"  Members: {channel.get('num_members', 'Unknown')}\n"
                if channel.get('topic', {}).get('value'):
                    output += f"  Topic: {channel['topic']['value']}\n"
                output += "\n"
            
            return output
            
        except Exception as e:
            return f"Error listing channels: {str(e)}"
    
    return _list_channels(types, exclude_archived, limit)


@server.tool("slack_list_users", description="List all users in the Slack workspace")
def slack_list_users(
    include_locale: Annotated[bool, Field(description="Include user locale information")] = False,
    limit: Annotated[int, Field(description="Maximum number of users to return", ge=1, le=1000)] = 100,
) -> str:
    """List all users in the Slack workspace."""
    
    @handle_slack_operation
    def _list_users(client, include_locale, limit):
        if not client:
            mock_users = [
                {"id": "U1234567890", "name": "john.doe", "real_name": "John Doe", "is_bot": False, "is_admin": True},
                {"id": "U2345678901", "name": "jane.smith", "real_name": "Jane Smith", "is_bot": False, "is_admin": False},
                {"id": "U3456789012", "name": "slackbot", "real_name": "Slackbot", "is_bot": True, "is_admin": False},
            ]
            
            output = f"Slack Users (showing {len(mock_users)} users):\n\n"
            for user in mock_users:
                output += f"• {user['real_name']} (@{user['name']})\n"
                output += f"  ID: {user['id']}\n"
                output += f"  Type: {'Bot' if user['is_bot'] else 'User'}\n"
                output += f"  Admin: {'Yes' if user['is_admin'] else 'No'}\n\n"
            
            return output
        
        try:
            response = client.users_list(limit=limit)
            
            users = response.get('members', [])
            
            if not users:
                return "No users found in the workspace"
            
            output = f"Slack Users ({len(users)} users):\n\n"
            for user in users:
                if user.get('deleted', False):
                    continue
                    
                output += f"• {user.get('real_name', 'Unknown')} (@{user.get('name', 'unknown')})\n"
                output += f"  ID: {user['id']}\n"
                output += f"  Type: {'Bot' if user.get('is_bot', False) else 'User'}\n"
                output += f"  Admin: {'Yes' if user.get('is_admin', False) else 'No'}\n"
                
                if include_locale and user.get('locale'):
                    output += f"  Locale: {user['locale']}\n"
                
                if user.get('profile', {}).get('title'):
                    output += f"  Title: {user['profile']['title']}\n"
                
                output += "\n"
            
            return output
            
        except Exception as e:
            return f"Error listing users: {str(e)}"
    
    return _list_users(include_locale, limit)


@server.tool("slack_get_channel_info", description="Get detailed information about a specific channel")
def slack_get_channel_info(
    channel: Annotated[str, Field(description="Channel ID or name")],
    include_num_members: Annotated[bool, Field(description="Include member count")] = True,
) -> str:
    """Get detailed information about a specific channel."""
    
    @handle_slack_operation
    def _get_channel_info(client, channel, include_num_members):
        if not client:
            return f"Mock channel info for {channel}:\nName: general\nID: C1234567890\nType: Public\nMembers: 25\nTopic: General discussion"
        
        try:
            response = client.conversations_info(
                channel=channel,
                include_num_members=include_num_members
            )
            
            channel_info = response['channel']
            
            output = f"Channel Information:\n"
            output += f"Name: #{channel_info['name']}\n"
            output += f"ID: {channel_info['id']}\n"
            output += f"Type: {'Private' if channel_info.get('is_private', False) else 'Public'}\n"
            
            if include_num_members and 'num_members' in channel_info:
                output += f"Members: {channel_info['num_members']}\n"
            
            if channel_info.get('topic', {}).get('value'):
                output += f"Topic: {channel_info['topic']['value']}\n"
            
            if channel_info.get('purpose', {}).get('value'):
                output += f"Purpose: {channel_info['purpose']['value']}\n"
            
            if channel_info.get('created'):
                output += f"Created: {channel_info['created']}\n"
            
            return output
            
        except Exception as e:
            return f"Error getting channel info: {str(e)}"
    
    return _get_channel_info(channel, include_num_members)


@server.tool("slack_get_user_info", description="Get detailed information about a specific user")
def slack_get_user_info(
    user: Annotated[str, Field(description="User ID or username")],
    include_locale: Annotated[bool, Field(description="Include user locale information")] = False,
) -> str:
    """Get detailed information about a specific user."""
    
    @handle_slack_operation
    def _get_user_info(client, user, include_locale):
        if not client:
            return f"Mock user info for {user}:\nName: John Doe\nUsername: john.doe\nID: U1234567890\nType: User\nAdmin: Yes"
        
        try:
            response = client.users_info(
                user=user,
                include_locale=include_locale
            )
            
            user_info = response['user']
            
            output = f"User Information:\n"
            output += f"Name: {user_info.get('real_name', 'Unknown')}\n"
            output += f"Username: @{user_info.get('name', 'unknown')}\n"
            output += f"ID: {user_info['id']}\n"
            output += f"Type: {'Bot' if user_info.get('is_bot', False) else 'User'}\n"
            output += f"Admin: {'Yes' if user_info.get('is_admin', False) else 'No'}\n"
            
            if include_locale and user_info.get('locale'):
                output += f"Locale: {user_info['locale']}\n"
            
            profile = user_info.get('profile', {})
            if profile.get('title'):
                output += f"Title: {profile['title']}\n"
            
            if profile.get('email'):
                output += f"Email: {profile['email']}\n"
            
            if profile.get('phone'):
                output += f"Phone: {profile['phone']}\n"
            
            return output
            
        except Exception as e:
            return f"Error getting user info: {str(e)}"
    
    return _get_user_info(user, include_locale)


@server.tool("slack_list_messages", description="List messages from a channel or conversation")
def slack_list_messages(
    channel: Annotated[str, Field(description="Channel ID or name")],
    limit: Annotated[int, Field(description="Maximum number of messages to return", ge=1, le=1000)] = 10,
    oldest: Annotated[str, Field(description="Start of time range (timestamp)")] = "",
    latest: Annotated[str, Field(description="End of time range (timestamp)")] = "",
    inclusive: Annotated[bool, Field(description="Include messages with latest or oldest timestamp")] = True,
) -> str:
    """List messages from a channel or conversation."""
    
    @handle_slack_operation
    def _list_messages(client, channel, limit, oldest, latest, inclusive):
        if not client:
            mock_messages = [
                {"text": "Hello everyone!", "user": "U1234567890", "ts": "1640995200.000100", "type": "message"},
                {"text": "How's the project going?", "user": "U2345678901", "ts": "1640995260.000200", "type": "message"},
            ]
            
            output = f"Recent Messages from #{channel} (showing {len(mock_messages)} messages):\n\n"
            for msg in mock_messages:
                output += f"• {msg['text']}\n"
                output += f"  User: {msg['user']}\n"
                output += f"  Time: {msg['ts']}\n\n"
            
            return output
        
        try:
            kwargs = {
                "channel": channel,
                "limit": limit,
                "inclusive": inclusive
            }
            
            if oldest:
                kwargs["oldest"] = oldest
            if latest:
                kwargs["latest"] = latest
            
            response = client.conversations_history(**kwargs)
            
            messages = response.get('messages', [])
            
            if not messages:
                return f"No messages found in #{channel}"
            
            output = f"Recent Messages from #{channel} ({len(messages)} messages):\n\n"
            for msg in messages:
                if msg.get('type') == 'message' and not msg.get('subtype'):
                    output += f"• {msg.get('text', 'No text')}\n"
                    output += f"  User: {msg.get('user', 'Unknown')}\n"
                    output += f"  Time: {msg.get('ts', 'Unknown')}\n"
                    if msg.get('thread_ts'):
                        output += f"  Thread: Yes\n"
                    output += "\n"
            
            return output
            
        except Exception as e:
            return f"Error listing messages: {str(e)}"
    
    return _list_messages(channel, limit, oldest, latest, inclusive)


@server.tool("slack_upload_file", description="Upload a file to a Slack channel")
def slack_upload_file(
    channel: Annotated[str, Field(description="Channel ID or name to upload to")],
    file_path: Annotated[str, Field(description="Local file path to upload")],
    title: Annotated[str, Field(description="File title")] = "",
    initial_comment: Annotated[str, Field(description="Initial comment with the file")] = "",
    thread_ts: Annotated[str, Field(description="Timestamp of parent message for thread")] = "",
) -> str:
    """Upload a file to a Slack channel."""
    
    @handle_slack_operation
    def _upload_file(client, channel, file_path, title, initial_comment, thread_ts):
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        if not client:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            return (
                f"Mock file upload successful!\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Size: {file_size_mb:.2f} MB\n"
                f"Channel: {channel}\n"
                f"Note: Install slack-sdk and set SLACK_BOT_TOKEN for real uploads."
            )
        
        try:
            upload_kwargs = {
                "channels": channel,
                "file": file_path
            }
            
            if title:
                upload_kwargs["title"] = title
            
            if initial_comment:
                upload_kwargs["initial_comment"] = initial_comment
            
            if thread_ts:
                upload_kwargs["thread_ts"] = thread_ts
            
            response = client.files_upload(**upload_kwargs)
            
            file_info = response['file']
            
            return (
                f"✅ File uploaded successfully!\n"
                f"File: {file_info.get('name', 'Unknown')}\n"
                f"Title: {file_info.get('title', 'No title')}\n"
                f"Size: {file_info.get('size', 0)} bytes\n"
                f"Channel: {channel}\n"
                f"File ID: {file_info['id']}\n"
                f"URL: {file_info.get('url_private', 'N/A')}"
            )
            
        except Exception as e:
            return f"Error uploading file: {str(e)}"
    
    return _upload_file(channel, file_path, title, initial_comment, thread_ts)


@server.tool("slack_create_channel", description="Create a new Slack channel")
def slack_create_channel(
    name: Annotated[str, Field(description="Channel name (without #)")],
    is_private: Annotated[bool, Field(description="Create as private channel")] = False,
    topic: Annotated[str, Field(description="Channel topic")] = "",
    purpose: Annotated[str, Field(description="Channel purpose")] = "",
) -> str:
    """Create a new Slack channel."""
    
    @handle_slack_operation
    def _create_channel(client, name, is_private, topic, purpose):
        if not client:
            return (
                f"Mock channel creation successful!\n"
                f"Name: #{name}\n"
                f"Type: {'Private' if is_private else 'Public'}\n"
                f"Note: Install slack-sdk and set SLACK_BOT_TOKEN for real channel creation."
            )
        
        try:
            create_kwargs = {
                "name": name,
                "is_private": is_private
            }
            
            response = client.conversations_create(**create_kwargs)
            
            channel = response['channel']
            
            # Set topic and purpose if provided
            if topic:
                client.conversations_setTopic(channel=channel['id'], topic=topic)
            
            if purpose:
                client.conversations_setPurpose(channel=channel['id'], purpose=purpose)
            
            return (
                f"✅ Channel created successfully!\n"
                f"Name: #{channel['name']}\n"
                f"ID: {channel['id']}\n"
                f"Type: {'Private' if channel.get('is_private', False) else 'Public'}\n"
                f"Topic: {topic if topic else 'No topic set'}\n"
                f"Purpose: {purpose if purpose else 'No purpose set'}"
            )
            
        except Exception as e:
            return f"Error creating channel: {str(e)}"
    
    return _create_channel(name, is_private, topic, purpose)


@server.tool("slack_invite_to_channel", description="Invite users to a channel")
def slack_invite_to_channel(
    channel: Annotated[str, Field(description="Channel ID or name")],
    users: Annotated[List[str], Field(description="List of user IDs to invite")],
) -> str:
    """Invite users to a channel."""
    
    @handle_slack_operation
    def _invite_to_channel(client, channel, users):
        if not client:
            return (
                f"Mock invitation successful!\n"
                f"Channel: {channel}\n"
                f"Invited users: {', '.join(users)}\n"
                f"Note: Install slack-sdk and set SLACK_BOT_TOKEN for real invitations."
            )
        
        try:
            response = client.conversations_invite(
                channel=channel,
                users=','.join(users)
            )
            
            channel_info = response['channel']
            
            return (
                f"✅ Users invited successfully!\n"
                f"Channel: #{channel_info['name']}\n"
                f"Invited users: {', '.join(users)}\n"
                f"Channel ID: {channel_info['id']}"
            )
            
        except Exception as e:
            return f"Error inviting users: {str(e)}"
    
    return _invite_to_channel(channel, users)


@server.tool("slack_set_channel_topic", description="Set the topic for a channel")
def slack_set_channel_topic(
    channel: Annotated[str, Field(description="Channel ID or name")],
    topic: Annotated[str, Field(description="New channel topic")],
) -> str:
    """Set the topic for a channel."""
    
    @handle_slack_operation
    def _set_channel_topic(client, channel, topic):
        if not client:
            return f"Mock topic set for {channel}: {topic}"
        
        try:
            response = client.conversations_setTopic(
                channel=channel,
                topic=topic
            )
            
            return (
                f"✅ Channel topic updated successfully!\n"
                f"Channel: {channel}\n"
                f"New topic: {topic}"
            )
            
        except Exception as e:
            return f"Error setting channel topic: {str(e)}"
    
    return _set_channel_topic(channel, topic)


@server.tool("slack_set_channel_purpose", description="Set the purpose for a channel")
def slack_set_channel_purpose(
    channel: Annotated[str, Field(description="Channel ID or name")],
    purpose: Annotated[str, Field(description="New channel purpose")],
) -> str:
    """Set the purpose for a channel."""
    
    @handle_slack_operation
    def _set_channel_purpose(client, channel, purpose):
        if not client:
            return f"Mock purpose set for {channel}: {purpose}"
        
        try:
            response = client.conversations_setPurpose(
                channel=channel,
                purpose=purpose
            )
            
            return (
                f"✅ Channel purpose updated successfully!\n"
                f"Channel: {channel}\n"
                f"New purpose: {purpose}"
            )
            
        except Exception as e:
            return f"Error setting channel purpose: {str(e)}"
    
    return _set_channel_purpose(channel, purpose)


@server.tool("slack_archive_channel", description="Archive a channel")
def slack_archive_channel(
    channel: Annotated[str, Field(description="Channel ID or name to archive")],
) -> str:
    """Archive a channel."""
    
    @handle_slack_operation
    def _archive_channel(client, channel):
        if not client:
            return f"Mock channel archived: {channel}"
        
        try:
            response = client.conversations_archive(channel=channel)
            
            return f"✅ Channel archived successfully: {channel}"
            
        except Exception as e:
            return f"Error archiving channel: {str(e)}"
    
    return _archive_channel(channel)


@server.tool("slack_unarchive_channel", description="Unarchive a channel")
def slack_unarchive_channel(
    channel: Annotated[str, Field(description="Channel ID or name to unarchive")],
) -> str:
    """Unarchive a channel."""
    
    @handle_slack_operation
    def _unarchive_channel(client, channel):
        if not client:
            return f"Mock channel unarchived: {channel}"
        
        try:
            response = client.conversations_unarchive(channel=channel)
            
            return f"✅ Channel unarchived successfully: {channel}"
            
        except Exception as e:
            return f"Error unarchiving channel: {str(e)}"
    
    return _unarchive_channel(channel)


@server.tool("slack_list_files", description="List files in the Slack workspace")
def slack_list_files(
    channel: Annotated[str, Field(description="Channel ID to list files from (optional)")] = "",
    user: Annotated[str, Field(description="User ID to list files from (optional)")] = "",
    types: Annotated[str, Field(description="File types: images, gdocs, zips, pdfs, etc.")] = "all",
    count: Annotated[int, Field(description="Number of files to return", ge=1, le=1000)] = 20,
) -> str:
    """List files in the Slack workspace."""
    
    @handle_slack_operation
    def _list_files(client, channel, user, types, count):
        if not client:
            mock_files = [
                {"id": "F1234567890", "name": "document.pdf", "title": "Important Document", "size": 1024000, "user": "U1234567890"},
                {"id": "F2345678901", "name": "image.png", "title": "Screenshot", "size": 512000, "user": "U2345678901"},
            ]
            
            output = f"Slack Files (showing {len(mock_files)} files):\n\n"
            for file in mock_files:
                output += f"• {file['title']}\n"
                output += f"  Name: {file['name']}\n"
                output += f"  ID: {file['id']}\n"
                output += f"  Size: {file['size']} bytes\n"
                output += f"  User: {file['user']}\n\n"
            
            return output
        
        try:
            kwargs = {
                "count": count,
                "types": types
            }
            
            if channel:
                kwargs["channel"] = channel
            if user:
                kwargs["user"] = user
            
            response = client.files_list(**kwargs)
            
            files = response.get('files', [])
            
            if not files:
                return "No files found"
            
            output = f"Slack Files ({len(files)} files):\n\n"
            for file in files:
                output += f"• {file.get('title', file.get('name', 'Unknown'))}\n"
                output += f"  Name: {file.get('name', 'Unknown')}\n"
                output += f"  ID: {file['id']}\n"
                output += f"  Size: {file.get('size', 0)} bytes\n"
                output += f"  User: {file.get('user', 'Unknown')}\n"
                output += f"  Type: {file.get('filetype', 'Unknown')}\n"
                if file.get('url_private'):
                    output += f"  URL: {file['url_private']}\n"
                output += "\n"
            
            return output
            
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    return _list_files(channel, user, types, count)


@server.tool("slack_get_workspace_info", description="Get information about the Slack workspace")
def slack_get_workspace_info() -> str:
    """Get information about the Slack workspace."""
    
    @handle_slack_operation
    def _get_workspace_info(client):
        if not client:
            return (
                "Mock workspace info:\n"
                "Name: My Workspace\n"
                "ID: T1234567890\n"
                "Domain: myworkspace\n"
                "Note: Install slack-sdk and set SLACK_BOT_TOKEN for real workspace info."
            )
        
        try:
            # Get team info
            team_response = client.team_info()
            team_info = team_response['team']
            
            # Get auth info
            auth_response = client.auth_test()
            
            output = f"Workspace Information:\n"
            output += f"Name: {team_info.get('name', 'Unknown')}\n"
            output += f"ID: {team_info.get('id', 'Unknown')}\n"
            output += f"Domain: {team_info.get('domain', 'Unknown')}\n"
            output += f"URL: {team_info.get('url', 'Unknown')}\n"
            output += f"Plan: {team_info.get('plan', 'Unknown')}\n"
            
            if team_info.get('icon'):
                output += f"Icon: {team_info['icon']}\n"
            
            output += f"\nAuthentication Info:\n"
            output += f"Bot User: {auth_response.get('user', 'Unknown')}\n"
            output += f"Team: {auth_response.get('team', 'Unknown')}\n"
            output += f"Team ID: {auth_response.get('team_id', 'Unknown')}\n"
            
            return output
            
        except Exception as e:
            return f"Error getting workspace info: {str(e)}"
    
    return _get_workspace_info()


# Static resource
@server.resource("config://slack_settings")
def get_slack_settings():
    return {
        "service_available": SLACK_AVAILABLE,
        "api_version": "v1",
        "scopes": [
            "channels:read",
            "channels:write",
            "chat:write",
            "files:read",
            "files:write",
            "groups:read",
            "groups:write",
            "im:read",
            "im:write",
            "mpim:read",
            "mpim:write",
            "users:read",
            "team:read"
        ],
        "supported_file_types": ["images", "gdocs", "zips", "pdfs", "spreadsheets", "presentations", "text", "videos", "audio"],
        "max_message_length": 40000,
        "max_file_size": "1GB",
        "available_tools": [
            "slack_send_message", "slack_list_channels", "slack_list_users", "slack_get_channel_info",
            "slack_get_user_info", "slack_list_messages", "slack_upload_file", "slack_create_channel",
            "slack_invite_to_channel", "slack_set_channel_topic", "slack_set_channel_purpose",
            "slack_archive_channel", "slack_unarchive_channel", "slack_list_files", "slack_get_workspace_info"
        ],
        "features": [
            "messaging", "channel_management", "user_management", "file_sharing",
            "threading", "rich_formatting", "bot_integration", "workflow_automation"
        ]
    }


# Dynamic resource template
@server.resource("slack://channel/{channel_id}/info")
def get_channel_info(channel_id: str):
    return {
        "channel_id": channel_id,
        "note": "Use slack_get_channel_info or other tools to get actual channel information",
    }


@server.prompt()
def slack_workflow_prompt(workflow_type: str) -> str:
    """Generate prompts for Slack workflows."""
    prompts = {
        "team_communication": "For team communication workflow:\n1. Use slack_list_channels to see available channels\n2. Use slack_send_message for announcements\n3. Use slack_create_channel for project-specific discussions\n4. Use slack_invite_to_channel to add team members\n5. Use slack_set_channel_topic and slack_set_channel_purpose for organization",
        "file_sharing": "For file sharing workflow:\n1. Use slack_upload_file to share documents\n2. Use slack_list_files to find shared files\n3. Use slack_send_message with attachments for context\n4. Organize files by channel for better management",
        "project_management": "For project management workflow:\n1. Create dedicated channels with slack_create_channel\n2. Set clear topics and purposes with slack_set_channel_topic/purpose\n3. Use slack_send_message for updates and discussions\n4. Use slack_upload_file for project documents\n5. Use slack_invite_to_channel to add stakeholders",
        "user_management": "For user management workflow:\n1. Use slack_list_users to see workspace members\n2. Use slack_get_user_info for detailed user information\n3. Use slack_invite_to_channel to add users to relevant channels\n4. Use slack_list_messages to monitor activity",
        "automation": "For automation workflow:\n1. Use slack_send_message for automated notifications\n2. Use slack_upload_file for automated reports\n3. Use slack_list_messages to monitor and respond to messages\n4. Use slack_get_workspace_info for system monitoring",
    }

    return prompts.get(workflow_type, f"Slack workflow guidance for: {workflow_type}")


from clarifai.runners.models.mcp_class import MCPModelClass


class SlackModelClass(MCPModelClass):
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

