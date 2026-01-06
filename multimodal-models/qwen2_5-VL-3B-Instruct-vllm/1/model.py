import os
import sys

sys.path.append(os.path.dirname(__file__))
from typing import Iterator, List

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_types import Image
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger
from openai import OpenAI


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

class VLLMModel(OpenAIModelClass):
    """
    A custom runner that integrates with the Clarifai platform and uses Server inference
    to process inputs, including text and images.
    """

    client = True  # This will be set in load_model method
    model = True  # This will be set in load_model method

    def load_model(self):
        """Load the model here and start the server."""
        os.path.join(os.path.dirname(__file__))

        server_args = {
            'max_model_len': '8192',
            'gpu_memory_utilization': 0.90,
            'dtype': 'auto',
            'task': 'auto',
            'kv_cache_dtype': 'auto',
            'tensor_parallel_size': 1,
            'chat_template': None,
            'cpu_offload_gb': 0.0,
            'quantization': None,
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

        logger.info(f"OpenAI {self.model} model loaded successfully!")

    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                image: Image = None,
                images: List[Image] = None,
                chat_history: List[dict] = None,
                max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", )
                ) -> str:
        """This is the method that will be called when the runner is run. It takes in an input and
        returns an output.
        """
        openai_messages = build_openai_messages(prompt=prompt, image=image, images=images, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
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
                max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", )
                ) -> Iterator[str]:
        """Example yielding a whole batch of streamed stuff back."""
        openai_messages = build_openai_messages(prompt=prompt, image=image, images=images, messages=chat_history)
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True):
            if chunk.choices:
                text = (chunk.choices[0].delta.content
                        if (chunk and chunk.choices[0].delta.content) is not None else '')
                yield text
                