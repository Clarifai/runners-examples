import os
import sys

sys.path.append(os.path.dirname(__file__))
from typing import List, Iterator

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger
from clarifai.runners.utils.data_types import Image

from openai import OpenAI
from openai_server_starter import OpenAI_APIServer

class DeepSeekOCRModel(OpenAIModelClass):
    """
    A Model that integrates with the Clarifai platform and uses SGlang framework for inference to run the DeepSeek-OCR multimodal model.
    """
    client = True  # This will be set in load_model method
    model = True  # This will be set in load_model method

    def load_model(self):
        """Load the model here and start the SGlang server."""
        os.path.join(os.path.dirname(__file__))

        # Vanilla SGlang server args - minimal configuration
        server_args = {
            'port': 23333,
            'host': '0.0.0.0',
            'checkpoints': 'deepseek-ai/DeepSeek-OCR'
        }

        # Start SGlang server
        self.server = OpenAI_APIServer.from_sglang_backend(**server_args)

        # Client initialization
        self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1'
        )
        self.model = self._get_model()

        logger.info(f"OpenAI {self.model} model loaded successfully!")

    def _get_model(self):
        try:
            return self.client.models.list().data[0].id
        except Exception as e:
            raise ConnectionError("Failed to retrieve model ID from API") from e

    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                image: Image = None,
                images: List[Image] = None,
                chat_history: List[dict] = None,
                tools: List[dict] = None,
                tool_choice: str = None,
                max_tokens: int = Param(default=4096, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass."),
                ) -> str:
        """
        This method is used to predict the response for the given prompt and chat history using the model and tools.
        """
        if tools is not None and tool_choice is None:
            tool_choice = "auto"
        messages = build_openai_messages(prompt=prompt, image=image, images=images, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p)

        if response.usage and response.usage.prompt_tokens and response.usage.completion_tokens:
            self.set_output_context(prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens)
        
        return response.choices[0].message.content


    @OpenAIModelClass.method
    def generate(self,
                prompt: str,
                image: Image = None,
                images: List[Image] = None,
                chat_history: List[dict] = None,
                tools: List[dict] = None,
                tool_choice: str = None,
                max_tokens: int = Param(default=4096, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.")) -> Iterator[str]:
        """
        This method is used to stream generated text tokens from a prompt + optional chat history and tools.
        """
        messages = build_openai_messages(prompt=prompt, image=image, images=images, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True)

        for chunk in response:
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                if hasattr(chunk.usage, 'prompt_tokens') and hasattr(chunk.usage, 'completion_tokens'):
                    self.set_output_context(prompt_tokens=chunk.usage.prompt_tokens, completion_tokens=chunk.usage.completion_tokens)
            if chunk.choices:
                text = (chunk.choices[0].delta.content
                        if (chunk and chunk.choices[0].delta.content) is not None else '')
                yield text