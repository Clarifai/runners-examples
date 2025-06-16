import os
import sys
from typing import Iterator, List

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger
from openai import OpenAI

PYTHON_EXEC = sys.executable

def from_sglang_backend(checkpoints, **kwargs):
    """Start SGlang OpenAI compatible server."""
    
    from sglang.utils import execute_shell_command, wait_for_server
    # Start building the command
    cmds = [
        PYTHON_EXEC, '-m', 'sglang.launch_server',
        '--model-path', checkpoints,
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
        'backend': "sglang",
        'process': None
    })
    
    try:
        server.process = execute_shell_command(" ".join(cmds))
        logger.info("Waiting for " + f"http://{server.host}:{server.port}")
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info("Server started successfully at " + f"http://{server.host}:{server.port}")
    except Exception as e:
        logger.error(f"Failed to start sglang server: {str(e)}")
        if server.process:
            server.process.terminate()
        raise RuntimeError(f"Failed to start sglang server: {str(e)}")

    return server

class SglangModel(OpenAIModelClass):
    """
    A custom runner that integrates with the Clarifai platform and uses Server inference
    to process inputs, including text.
    """

    client = True  # This will be set in load_model method
    model = True  # This will be set in load_model method

    def load_model(self):
        """Load the model here and start the  server."""
        # server args for sglang
        # You can change these parameters according to your needs
        server_args = {
                    'dtype': 'auto',
                    'kv_cache_dtype': 'auto',
                    'tp_size': 1,
                    'load_format': 'auto',
                    'context_length': None,
                    'device': 'cuda',
                    'port': 23333,
                    'host': '0.0.0.0',
                    'mem_fraction_static': 0.9,
                    'max_total_tokens': '8192',
                    'max_prefill_tokens': None,
                    'schedule_policy': 'fcfs',
                    'schedule_conservativeness': 1.0,
                    'trust_remote_code': True,
                    }
        
        checkpoints = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        # Start server
        self.server = from_sglang_backend(checkpoints, **server_args)

        # Create client
        self.client = OpenAI(
                api_key="notset",
                base_url= f"http://{self.server.host}:{self.server.port}/v1")
        self.model = self.client.models.list().data[0].id

        logger.info(f"OpenAI {self.model} model loaded successfully!")

    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                chat_history: List[dict] = None,
                max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.8, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", )
                ) -> str:
        """This is the method that will be called when the runner is run. It takes in an input and
        returns an output.
        """
        openai_messages = build_openai_messages(prompt=prompt, messages=chat_history)
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
                chat_history: List[dict] = None,
                max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.8, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", )
                ) -> Iterator[str]:
        """Example yielding a whole batch of streamed stuff back."""
        openai_messages = build_openai_messages(prompt=prompt, messages=chat_history)
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

    # This method is needed to test the model with the test-locally CLI command.
    def test(self):
        """Test the model here."""
        try:
            print("Testing predict...")
            # Test predict
            print(self.predict(prompt="Hello, how are you?",))
        except Exception as e:
            print("Error in predict", e)

        try:
            print("Testing generate...")
            # Test generate
            for each in self.generate(prompt="Hello, how are you?",):
                print(each, end=" ")
        except Exception as e:
            print("Error in generate", e)