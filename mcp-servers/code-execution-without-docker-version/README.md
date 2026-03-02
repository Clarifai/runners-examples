---
complexity: advanced
framework: fastmcp
model_size: N/A
gpu_required: false
min_gpu_memory: N/A
features: [mcp, code-execution]
model_class: StdioMCPModelClass
task: mcp-server
---


# Code Execution MCP Tool

This MCP tool provides Python code execution capabilities using the local Python environment. It allows executing Python code with extra packages installed if needed.

## Setup Instructions

This tool can support both local-runner and deployment:

1. For local development: `clarifai model local-runner`
2. For deployment: `clarifai model upload`