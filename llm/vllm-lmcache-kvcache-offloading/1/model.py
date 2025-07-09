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

def vllm_openai_server(checkpoints, lmcache_config=None, **kwargs):
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
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                # likely JSON: wrap in single quotes
                cmds.extend([f'--{param_name}', f"'{value}'"])
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
        if lmcache_config:
            for k, v in lmcache_config.items():
                os.environ[k] = v
        print(f'Starting vLLM server with command: {" ".join(cmds)}')
        server.process = execute_shell_command(" ".join(cmds))
        logger.info("Waiting for " + f"http://{server.host}:{server.port}")
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info("Server started successfully at " + f"http://{server.host}:{server.port}")
    except Exception as e:
        logger.error(f"Failed to start vllm server: {str(e)}")
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start vllm server: {str(e)}")
    finally:
        if lmcache_config:
            for k in lmcache_config.keys():
                if k in os.environ:
                    del os.environ[k]
    return server

class VLLMLlamaModel(OpenAIModelClass):
  """
  A Model that integrates with the Clarifai platform and uses vLLM framework for inference to run the Llama 3.1 8B model.
  """
  client = True  # This will be set in load_model method
  model = True  # This will be set in load_model method

  def load_model(self):
    """Load the model here and start the  server."""
    server_args = {
        'gpu_memory_utilization': 0.9,
        'dtype': 'auto',
        'task': 'auto',
        'kv_cache_dtype': 'auto',
        'tensor_parallel_size': 1,
        'quantization': None,
        'cpu_offload_gb': 0.0,
        'port': 23333,
        'host': 'localhost',
        # kv_transfer_config: This is the parameter that actually tells vLLM to use LMCache for KV cache offloading.
        'kv_transfer_config': '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}', 
    }

    model_path = os.path.dirname(os.path.dirname(__file__))
    builder = ModelBuilder(model_path, download_validation_only=True)
    model_config = builder.config
    
    stage = model_config["checkpoints"]['when']
    checkpoints = builder.config["checkpoints"]['repo_id']
    if stage in ["build", "runtime"]:
      checkpoints = builder.download_checkpoints(stage=stage)

    # Start server
    lmcache_config = {
        'LMCACHE_USE_EXPERIMENTAL': 'True',
        'LMCACHE_CHUNK_SIZE': '256',  # 256 Tokens per KV Chunk
        'LMCACHE_LOCAL_CPU': 'True', # Enable CPU memory backend
        'LMCACHE_MAX_LOCAL_CPU_SIZE': '5.0',  # 5GB of Pinned CPU memory
    }
    self.server = vllm_openai_server(checkpoints, lmcache_config=lmcache_config, **server_args)
    # CLIent initialization
    self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1')
    self.model = self.client.models.list().data[0].id

  @OpenAIModelClass.method
  def predict(self,
              prompt: str,
              chat_history: List[dict] = None,
              max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
              temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
              top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass."), 
              ) -> str:
    """
    This method is used to predict the response for the given prompt and chat history using the model and tools.
    """
            
    messages = build_openai_messages(prompt=prompt, messages=chat_history)
    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p)
      
    if response.choices[0]:
      return response.choices[0].message.content
    

  @OpenAIModelClass.method
  def generate(self,
               prompt: str,
               chat_history: List[dict] = None,
               max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
               temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
               top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.")) -> Iterator[str]:
    """
    This method is used to stream generated text tokens from a prompt + optional chat history and tools.
    """
    messages = build_openai_messages(prompt=prompt, messages=chat_history)
    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True)
    
    for chunk in response:
      if chunk.choices:
        # Otherwise, return the content of the first choice
        text = (chunk.choices[0].delta.content
                if (chunk and chunk.choices[0].delta.content) is not None else '')
        yield text