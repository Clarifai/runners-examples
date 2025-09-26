# Code Execution MCP Tool

This MCP tool provides Python code execution capabilities in Docker containers. It allows executing Python code with package installation support.

## Setup Instructions

1. Ensure the Docker socket path in `1/model.py` (around line 31) points to the correct Docker socket on your local machine
2. Make sure Docker daemon is running
3. Run: `clarifai model upload local-runner`