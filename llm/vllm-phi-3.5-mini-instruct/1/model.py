import os
import sys
from typing import List, Iterator

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from openai import OpenAI
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger

PYTHON_EXEC = sys.executable

def vllm_openai_server(checkpoints, **kwargs):
    """Start vLLM OpenAI compatible server."""

    from clarifai.runners.utils.model_utils import execute_shell_command, wait_for_server, terminate_process

    cmds = [
        PYTHON_EXEC, '-m', 'vllm.entrypoints.openai.api_server', '--model', checkpoints,
    ]

    for key, value in kwargs.items():
        if value is None:
            continue
        param_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmds.append(f'--{param_name}')
        else:
            cmds.extend([f'--{param_name}', str(value)])

    server = type('Server', (), {
        'host': kwargs.get('host', '0.0.0.0'),
        'port': kwargs.get('port', 23333),
        'backend': "vllm",
        'process': None
    })()

    try:
        server.process = execute_shell_command(" ".join(cmds))
        logger.info(f"Waiting for http://{server.host}:{server.port}")
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info(f"Server started successfully at http://{server.host}:{server.port}")
    except Exception as e:
        logger.error(f"Failed to start vllm server: {str(e)}")
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start vllm server: {str(e)}")

    return server


class Phi35MiniModel(OpenAIModelClass):
    """
    Microsoft Phi-3.5-mini-instruct model with vLLM backend.
    """
    client = True
    model = True

    def load_model(self):
        """Load the model and start the vLLM server."""

        server_args = {
            'max_model_len': 4096,
            'gpu_memory_utilization': 0.9,
            'dtype': 'auto',
            'task': 'auto',
            'kv_cache_dtype': 'auto',
            'tensor_parallel_size': 1,
            'port': 23333,
            'host': 'localhost',
            'trust_remote_code': True,  # Required for Phi models
        }

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        model_config = builder.config

        stage = model_config["checkpoints"]['when']
        checkpoints = builder.config["checkpoints"]['repo_id']
        if stage in ["build", "runtime"]:
            checkpoints = builder.download_checkpoints(stage=stage)

        # Start vLLM server
        self.server = vllm_openai_server(checkpoints, **server_args)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1'
        )
        self.model = self.client.models.list().data[0].id
        logger.info(f"Phi-3.5-mini model loaded successfully: {self.model}")

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str,
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate."
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (0-2)."
        ),
        top_p: float = Param(
            default=0.95,
            description="Nucleus sampling probability."
        ),
    ) -> str:
        """Generate text response for the given prompt."""

        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        if response.usage:
            self.set_output_context(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str,
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate."
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (0-2)."
        ),
        top_p: float = Param(
            default=0.95,
            description="Nucleus sampling probability."
        ),
    ) -> Iterator[str]:
        """Stream generated text tokens."""

        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )

        for chunk in response:
            if chunk.choices:
                text = (
                    chunk.choices[0].delta.content
                    if chunk.choices[0].delta.content is not None
                    else ''
                )
                yield text

    def test(self):
        """Test the model."""
        try:
            print("Testing predict...")
            result = self.predict(prompt="What is the capital of France?")
            print(result)
        except Exception as e:
            print(f"Error in predict: {e}")

        try:
            print("\nTesting generate (streaming)...")
            for chunk in self.generate(prompt="Explain quantum computing in simple terms."):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"Error in generate: {e}")
