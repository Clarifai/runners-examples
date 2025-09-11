# Slack MCP Server

A comprehensive Model Control Protocol (MCP) server for Slack operations, providing full messaging, channel management, user management, file sharing, and advanced team collaboration capabilities.

## Configuration

Must have `SLACK_BOT_TOKEN` in your secrets! and use `secrets` section in `config.yaml`

```
secrets:
  - id: "slack_key"
    type: "env"
    env_var: "SLACK_BOT_TOKEN"
    description: "API key of Slack for messaging and channel management"
```


### Deployment to Clarifai

The server includes a `config.yaml` file for deployment to Clarifai's model hosting platform:

```bash
clarifai model upload
```

## usage

Use `client.py` for MCP tool call


## 🚀 Features

### Core Messaging
- **Message Sending**: Send messages to channels, users, or threads
- **Rich Formatting**: Support for Slack blocks and attachments
- **Threading**: Reply to messages in threads
- **Message History**: Retrieve and list message history

### Channel Management
- **Channel Operations**: Create, archive, unarchive channels
- **Channel Information**: Get detailed channel information
- **Topic & Purpose**: Set and update channel topics and purposes
- **User Management**: Invite users to channels

### User Management
- **User Information**: Get detailed user profiles and information
- **User Lists**: List all workspace users with filtering
- **User Status**: Check user status and activity

### File Sharing
- **File Upload**: Upload files to channels with comments
- **File Management**: List and manage shared files
- **File Types**: Support for all Slack-supported file types

### Advanced Features
- **Workspace Information**: Get comprehensive workspace details
- **Bot Integration**: Full bot functionality and automation
- **Workflow Automation**: Automated messaging and file operations
- **Rich Content**: Support for blocks, attachments, and formatting

## 🛠️ Available Tools

### Messaging Tools
1. **`slack_send_message`** - Send messages to channels or users
2. **`slack_list_messages`** - List messages from channels or conversations

### Channel Management
3. **`slack_list_channels`** - List all channels in the workspace
4. **`slack_get_channel_info`** - Get detailed channel information
5. **`slack_create_channel`** - Create new channels (public or private)
6. **`slack_invite_to_channel`** - Invite users to channels
7. **`slack_set_channel_topic`** - Set channel topics
8. **`slack_set_channel_purpose`** - Set channel purposes
9. **`slack_archive_channel`** - Archive channels
10. **`slack_unarchive_channel`** - Unarchive channels

### User Management
11. **`slack_list_users`** - List all users in the workspace
12. **`slack_get_user_info`** - Get detailed user information

### File Management
13. **`slack_upload_file`** - Upload files to channels
14. **`slack_list_files`** - List files in the workspace

### Workspace Information
15. **`slack_get_workspace_info`** - Get workspace information

## 📋 Prerequisites

### Required Environment Variables
- **`CLARIFAI_PAT`**: Your Clarifai Personal Access Token
- **`SLACK_BOT_TOKEN`**: Your Slack Bot Token (for bot functionality)

### Slack App Setup
1. **Create a Slack App**:
   - Go to [Slack API](https://api.slack.com/apps)
   - Click "Create New App" → "From scratch"
   - Enter app name and select workspace

2. **Configure OAuth & Permissions**:
   - Go to "OAuth & Permissions"
   - Add the following Bot Token Scopes:
     - `channels:read` - View basic information about public channels
     - `channels:write` - Manage public channels
     - `chat:write` - Send messages as the bot
     - `files:read` - View files shared in channels
     - `files:write` - Upload files
     - `groups:read` - View basic information about private channels
     - `groups:write` - Manage private channels
     - `im:read` - View basic information about direct messages
     - `im:write` - Start direct messages
     - `mpim:read` - View basic information about group direct messages
     - `mpim:write` - Start group direct messages
     - `users:read` - View people in the workspace
     - `team:read` - View workspace information

3. **Install the App**:
   - Click "Install to Workspace"
   - Copy the "Bot User OAuth Token" (starts with `xoxb-`)

## 🔧 Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Slack Secrets in config.yaml**:



## 🔐 Security & Permissions

### Bot Token Scopes
The server requires comprehensive scopes for full functionality:
- **Channel Management**: `channels:read`, `channels:write`, `groups:read`, `groups:write`
- **Messaging**: `chat:write`, `im:read`, `im:write`, `mpim:read`, `mpim:write`
- **File Management**: `files:read`, `files:write`
- **User Management**: `users:read`
- **Workspace Info**: `team:read`

### Security Best Practices
1. **Use Bot Tokens** for automated operations
2. **Use User Tokens** for user-specific operations
3. **Limit token scopes** to only what's needed
4. **Rotate tokens** regularly
5. **Monitor token usage** in Slack workspace settings

## 📊 Configuration

### Slack Settings Resource
Access comprehensive configuration via:
```python
result = await client.read_resource("config://slack_settings")
```

### Available Features
- **File Types**: images, gdocs, zips, pdfs, spreadsheets, presentations, text, videos, audio
- **Max Message Length**: 40,000 characters
- **Max File Size**: 1GB
- **Supported Channels**: public, private, direct messages, group messages
