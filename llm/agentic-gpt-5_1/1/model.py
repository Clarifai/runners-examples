from typing import List, Iterator, Dict, Any
import asyncio
import os
import json

from clarifai.runners.models.agentic_class import AgenticModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.data_types import Image
from clarifai.runners.utils.openai_convertor import build_openai_messages
from openai import OpenAI
from clarifai.utils.logging import logger
from pydantic_core import from_json, to_json
from clarifai_grpc.grpc.api.status import status_code_pb2

# Set your OpenAI API key here
OPENAI_API_KEY = 'OPENAI_API_KEY'

class OPENAI_GPT_4o(AgenticModelClass):
    """
    A custom runner that integrates with the Clarifai platform and uses OpenAI gpt-4o model to process inputs, including text and images. This model also supports tool calling and streaming.
    """
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1/")
    model = "gpt-5.1-2025-11-13"
    
    def load_model(self):
        """Load the model here and start the server."""
        # log that system is ready
        logger.info(f"OpenAI {self.model} model loaded successfully!")
    
            
    @AgenticModelClass.method
    def predict(
        self,
        prompt: str,
        image: Image = None,
        images: List[Image] = None,
        tools: List[dict] = None,
        tool_choice: str = None,
        mcp_servers: List[str] = None,
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=4096, description="An upper bound for the number of tokens that can be generated for a completion."),
        temperature: float = Param(default=1.0, description="A decimal number that determines the degree of randomness in the response"),
        top_p: float = Param(default=1.0, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass."),
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
        image: Image = None,
        images: List[Image] = None,
        tools: List[dict] = None,
        tool_choice: str = None,
        mcp_servers: List[str] = None,
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=4096, description="An upper bound for the number of tokens that can be generated for a completion."),
        temperature: float = Param(default=1.0, description="A decimal number that determines the degree of randomness in the response"),
        top_p: float = Param(default=1.0, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass."),
    ) -> Iterator[str]:
        """Generate the output of the model using OpenAI-compatible API with optional MCP tool support."""
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