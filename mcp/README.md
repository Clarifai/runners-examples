# MCP Examples

This directory contains example MCP (Model Control Protocol) servers built with the FastMCP framework. Each example demonstrates how to create specialized MCP servers for different use cases.

## Directory Structure

Each MCP server example follows the same structure:
- `1/model.py` - The main MCP server implementation
- `config.yaml` - Model configuration for Clarifai deployment
- `requirements.txt` - Python dependencies
- `client.py` - Example client demonstrating usage

## Available Examples

### 1. Browser Tools (`browser-tools/`)
**Purpose**: Web browsing, scraping, and search capabilities
**Key Features**:
- Webpage content fetching
- HTML parsing and text extraction
- Link extraction and analysis
- Website status checking
- Search functionality (mock implementation)
- Content search within pages

### 2. Google Drive (`google-drive/`)
**Purpose**: Google Drive operations for file storage and collaboration
**Key Features**:
- File listing and search
- Upload and download operations
- Sharing and permission management
- Folder creation and organization
- File format conversion
- Collaboration features

### 3. PostgreSQL (`postgres/`)
**Purpose**: PostgreSQL database operations and advanced features
**Key Features**:
- Database connection and query execution
- Advanced PostgreSQL features (JSONB, arrays)
- Table statistics and analysis
- Connection monitoring
- Backup and restore operations

## Usage

### Running an Example

1. Navigate to the desired example directory:
```bash
cd runners-examples/mcp/browser-tools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the client example:
```bash
python client.py
```

### Deploying to Clarifai

Each example includes a `config.yaml` file for deployment to Clarifai's model hosting platform:

1. Ensure you have the Clarifai CLI installed
2. Configure your credentials
3. Deploy the model:
```bash
clarifai model upload
```

## Configuration

### Environment Variables

Many examples require specific environment variables for authentication:

- **PostgreSQL**: `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DATABASE`
- **Google Drive**: OAuth 2.0 credentials setup required

### Mock Data

Examples include mock data and fallback implementations when external services are not available, allowing you to test the MCP interface without requiring all external dependencies.

## Implementation Details

### FastMCP Framework

All examples use the FastMCP v2 framework which provides:
- Automatic tool registration from function definitions
- Type validation using Pydantic
- Resource and prompt management
- HTTP and stdio transport support

### Error Handling

Each MCP server implements comprehensive error handling:
- Input validation
- Service availability checks
- Graceful degradation with mock data
- Detailed error messages

### Security

Code execution examples (like code-sandbox) implement security measures:
- Restricted imports and builtins
- Execution timeouts
- Safe environment isolation
- Input sanitization

## Extending Examples

To create a new MCP server example:

1. Create a new directory following the naming convention
2. Implement the MCP server in `1/model.py`
3. Create appropriate `config.yaml`, `requirements.txt`, and `client.py` files
4. Follow the established patterns for tools, resources, and prompts
5. Include proper error handling and mock data support

## Contributing

When adding new examples:
- Follow the existing directory structure
- Include comprehensive documentation
- Provide working client examples
- Test with both real and mock data
- Ensure proper error handling
