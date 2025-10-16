# Code Execution MCP Tool with Docker

This MCP tool provides Python code execution capabilities in Docker containers using Clarifai Local Runner. It allows executing Python code with package installation support in isolated Docker containers.

## Prerequisites

1. **Docker**: Docker must be installed and running on your local machine
2. **Clarifai Local Runner**: Set up according to https://docs.clarifai.com/compute/local-runners/
3. **Docker Socket Access**: The MCP server needs access to your local Docker daemon

## Setup Instructions

### Step 1: Configure Docker Socket Path

**IMPORTANT**: Before deployment, you must configure the Docker socket path for your system.

1. Open `1/model.py`
2. Find the `get_docker_client()` function (around line 19)
3. Locate the `alternative_sockets` list (around line 47)
4. Update the first entry with your Docker socket path:

   ```python
   alternative_sockets = [
       # TODO: UPDATE THIS PATH - Replace with your actual Docker socket path
       'unix:///Users/YOUR_USERNAME/.rd/docker.sock',  # Update this
       ...
   ]
   ```
