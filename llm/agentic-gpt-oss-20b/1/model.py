import os
import sys

from typing import List, Iterator

from clarifai.runners.models.model_builder import ModelBuilder
from openai import OpenAI
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger
from clarifai.runners.models.agentic_class import AgenticModelClass
import json

PYTHON_EXEC = sys.executable

def vllm_openai_server(checkpoints, **kwargs):
    """Start vLLM OpenAI compatible server."""
    
    from clarifai.runners.utils.model_utils import execute_shell_command, wait_for_server, terminate_process
    # Start building the command
    cmds = [
        PYTHON_EXEC, '-m', 'vllm.entrypoints.openai.api_server', '--model', checkpoints,
    ]
    # Add all parameters from kwargs to the command
    for key, value in kwargs.items():
        if value is None:  # Skip None values
            continue
        param_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:  # Only add the flag if True
                cmds.append(f'--{param_name}')
        else:
            cmds.extend([f'--{param_name}', str(value)])
    # Create server instance
    server = type('Server', (), {
        'host': kwargs.get('host', '0.0.0.0'),
        'port': kwargs.get('port', 23333),
        'backend': "vllm",
        'process': None
    })()
    
    try:
        server.process = execute_shell_command(" ".join(cmds))
        logger.info("Waiting for " + f"http://{server.host}:{server.port}")
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info("Server started successfully at " + f"http://{server.host}:{server.port}")
    except Exception as e:
        logger.error(f"Failed to start vllm server: {str(e)}")
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start vllm server: {str(e)}")

    return server

class VLLMGPT_OSSModel(AgenticModelClass):
    """
    A Model that integrates with the Clarifai platform and uses vLLM framework for inference to run the GPT-OSS model.
    """
    client = True  # This will be set in load_model method
    model = True  # This will be set in load_model method

    def load_model(self):
    """Load the model here and start the  server."""
    os.path.join(os.path.dirname(__file__))

    server_args = {
        # 'max_model_len': 2048,
        'gpu_memory_utilization': 0.9,
        'dtype': 'auto',
        'task': 'auto',
        'kv_cache_dtype': 'auto',
        'tensor_parallel_size': 1,
        'port': 23333,
        'host': 'localhost',
    }

    model_path = os.path.dirname(os.path.dirname(__file__))
    builder = ModelBuilder(model_path, download_validation_only=True)
    model_config = builder.config

    stage = model_config["checkpoints"]['when']
    checkpoints = builder.config["checkpoints"]['repo_id']
    if stage in ["build", "runtime"]:
        checkpoints = builder.download_checkpoints(stage=stage)

    # Start server
    self.server = vllm_openai_server(checkpoints, **server_args)
    # CLIent initialization
    self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1')
    self.model = self.client.models.list().data[0].id

    @AgenticModelClass.method
    def predict(
        self,
        prompt: str,
        tools: List[dict] = None,
        tool_choice: str = None,
        mcp_servers: List[str] = None,
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=4096, description="An upper bound for the number of tokens that can be generated for a completion."),
        temperature: float = Param(default=1.0, description="A decimal number that determines the degree of randomness in the response"),
        top_p: float = Param(default=1.0, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass."),
        reasoning_effort: str = Param(
            default="medium",
            description="The level of reasoning effort to apply to the response. Currently supported values are low, medium, and high. ",
        ),
    ) -> str:
        """Predict the output of the model using OpenAI-compatible API with optional MCP tool support.
        """
        # Set default tool_choice if tools are provided
        if tools is not None and tool_choice is None:
            tool_choice = "auto"

        # Build OpenAI messages from prompt, images, and chat history
        openai_messages = build_openai_messages(prompt=prompt, image=image, images=images, messages=chat_history)

        # Build request data dictionary
        request_data = {
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "openai_endpoint": self.ENDPOINT_CHAT_COMPLETIONS,
        }

        # Add tools if provided (but not if MCP servers are provided, as they will discover tools)
        if tools is not None:
            request_data["tools"] = tools
            request_data["tool_choice"] = tool_choice
        elif mcp_servers is not None and len(mcp_servers) > 0:
            request_data["mcp_servers"] = mcp_servers
        request_json = json.dumps(request_data)
        response_json = self.openai_transport(request_json)
        try:
            response_data = json.loads(response_json)
            # Extract content from response
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content", "")
                return content
            else:
                raise ValueError("Failed to get response from the Model")
                
        except Exception as e:
            raise Exception(f"Failed to get response from the Model: {e}")

    @AgenticModelClass.method
    def generate(self,
        prompt: str,
        tools: List[dict] = None,
        tool_choice: str = None,
        mcp_servers: List[str] = None,
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=4096, description="An upper bound for the number of tokens that can be generated for a completion."),
        temperature: float = Param(default=1.0, description="A decimal number that determines the degree of randomness in the response"),
        top_p: float = Param(default=1.0, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass."),
        reasoning_effort: str = Param(
            default="medium",
            description="The level of reasoning effort to apply to the response. Currently supported values are low, medium, and high. ",
        ),
    ) -> Iterator[str]:
        """Generate the output of the model using OpenAI-compatible API with optional MCP tool support."""
        # Set default tool_choice if tools are provided
        if tools is not None and tool_choice is None:
            tool_choice = "auto"

        # Build request data dictionary
        request_data = {
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "openai_endpoint": self.ENDPOINT_CHAT_COMPLETIONS,
        }
        
        # Add tools if provided (but not if MCP servers are provided, as they will discover tools)
        if tools is not None:
            request_data["tools"] = tools
            request_data["tool_choice"] = tool_choice
        elif mcp_servers is not None and len(mcp_servers) > 0:
            request_data["mcp_servers"] = mcp_servers
        request_json = json.dumps(request_data)
        response_json = self.openai_stream_transport(request_json)
        for chunk in response_json:
            chunk_json = json.loads(chunk)
            if chunk_json and chunk_json.get("choices") and len(chunk_json["choices"]) > 0:
                choice = chunk_json["choices"][0]
                if choice and choice.get("delta") and choice["delta"].get("content"):
                    yield choice["delta"]["content"]
                elif choice and choice.get("delta") and choice["delta"].get("reasoning_content"):
                    yield choice["delta"]["reasoning_content"]
                else:
                    yield ""
        yield ""