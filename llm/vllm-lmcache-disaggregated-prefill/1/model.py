import os
import sys
from typing import List, Iterator

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from openai import OpenAI
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger
import subprocess
import shlex

PYTHON_EXEC = sys.executable

# Add these to your model.py or a utils module

import subprocess
import time
import os
import signal
import requests

def execute_shell_command(
    command: str,
    env=None
) -> subprocess.Popen:
    """Execute a shell command and return its process handle.

    Args:
        command (str): The shell command to execute.

    Returns:
        subprocess.Popen: Process handle for the executed command.

    Raises:
        ValueError: If command is empty or invalid.
        subprocess.SubprocessError: If command execution fails.
    """
    if not command or not isinstance(command, str):
        raise ValueError("command must be a non-empty string")

    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = shlex.split(command)

    try:
        print(f'parts: {parts}')
        process = subprocess.Popen(parts, env= env, text=True, stderr=subprocess.STDOUT)

        return process
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to execute command: {e}")
        raise

def start_vllm_disagg_server(
    checkpoints,
    prefiller_config: dict,
    decoder_config: dict,
    proxy_config: dict
):
    """
    Launch vLLM LMCache Disaggregated Prefill servers: Prefiller, Decoder, Proxy.
    Returns: dict (handles for cleanup & info)
    """
    server_handles = {}
    python_exec = sys.executable

    #####################
    # Start Decoder
    #####################
    decoder_env = os.environ.copy()
    decoder_env['UCX_TLS'] = 'cuda_ipc,cuda_copy,tcp'
    decoder_env['LMCACHE_CONFIG_FILE'] = decoder_config['config_file']
    decoder_env['CUDA_VISIBLE_DEVICES'] = str(decoder_config.get('cuda_visible_devices', 1))

    decoder_cmd = [
        python_exec, '-m', 'vllm.entrypoints.openai.api_server', '--model', checkpoints,
        '--port', str(decoder_config['port']),
        '--disable-log-requests',
        '--kv-transfer-config', f"'{decoder_config['kv_transfer_config']}'",
    ]

    print('Starting Disaggregated Decoder:', ' '.join(decoder_cmd))
    decoder_cmd_str = ' '.join(decoder_cmd)
    decoder_proc = execute_shell_command(decoder_cmd_str, env=decoder_env,)
    server_handles['decoder'] = {"proc": decoder_proc, "port": decoder_config['port']}
    # Wait for decoder to be up
    _wait_for_http_server('localhost', decoder_config['port'])

    #####################
    # Start Prefiller
    #####################
    prefiller_env = os.environ.copy()
    prefiller_env['UCX_TLS'] = 'cuda_ipc,cuda_copy,tcp'
    prefiller_env['LMCACHE_CONFIG_FILE'] = prefiller_config['config_file']
    prefiller_env['CUDA_VISIBLE_DEVICES'] = str(prefiller_config.get('cuda_visible_devices', 0))

    prefiller_cmd = [
        python_exec, '-m', 'vllm.entrypoints.openai.api_server', '--model', checkpoints,
        '--port', str(prefiller_config['port']),
        '--disable-log-requests',
        '--kv-transfer-config', f"'{prefiller_config['kv_transfer_config']}'",
    ]

    print('Starting Disaggregated Prefiller:', ' '.join(prefiller_cmd))
    prefiller_cmd_str = ' '.join(prefiller_cmd)
    prefiller_proc = execute_shell_command(prefiller_cmd_str, env=prefiller_env,)
    server_handles['prefiller'] = {"proc": prefiller_proc, "port": prefiller_config['port']}
    _wait_for_http_server('localhost', prefiller_config['port'])

    #####################
    # Start Proxy Server
    #####################
    # This assumes proxy server script is in same dir as model.py
    proxy_script = os.path.join(os.path.dirname(__file__), 'disagg_proxy_server.py')

    proxy_cmd = [
        python_exec, proxy_script,
        '--host', proxy_config['host'],
        '--port', str(proxy_config['port']),
        '--prefiller-host', proxy_config['prefiller_host'],
        '--prefiller-port', str(proxy_config['prefiller_port']),
        '--decoder-host', proxy_config['decoder_host'],
        '--decoder-port', str(proxy_config['decoder_port']),
    ]
    print('Starting Proxy:', ' '.join(proxy_cmd))
    proxy_cmd_str = ' '.join(proxy_cmd)
    proxy_proc = execute_shell_command(proxy_cmd_str,)
    server_handles['proxy'] = {"proc": proxy_proc, "port": proxy_config['port']}
    # Give the proxy some time to come up (or implement /health endpoint check if available)
    # _wait_for_http_server(proxy_config['host'], proxy_config['port'])

    return server_handles

def _wait_for_http_server(host, port, timeout=300):
    url = f'http://{host}:{port}/v1/models'
    for _ in range(timeout):
        try:
            r = requests.get(url, headers={"Authorization": "Bearer None"},)
            if r.status_code == 200:
                return
            else:
                print(f"response: {r}")
        except Exception as e:
            pass
            # print(e)
        time.sleep(1)
    raise RuntimeError(f"Server http://{host}:{port} didn't come up in {timeout} seconds!")

def stop_vllm_disagg_server(server_handles):
    for k, info in server_handles.items():
        proc = info['proc']
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as e:
            print(f"Warning: could not kill {k} process: {e}")


class VLLMLlamaModel(OpenAIModelClass):
  """
  A Model that integrates with the Clarifai platform and uses vLLM framework for inference to run the Llama 3.1 8B model.
  """
  client = True  # This will be set in load_model method
  model = True  # This will be set in load_model method

  def load_model(self):
        """Launch prefiller, decoder, proxy server and connect OpenAI client."""
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        model_config = builder.config

        stage = model_config["checkpoints"]["when"]
        checkpoints = model_config["checkpoints"]["repo_id"]
        if stage in ["build", "runtime"]:
            checkpoints = builder.download_checkpoints(stage=stage)
          
        # Disagg configs (point to your yamls and set correct ports)
        prefiller_config = {
            'config_file': os.path.join(os.path.dirname(__file__), 'lmcache-prefiller-config.yaml'),
            'port': 8100,
            'cuda_visible_devices': 0,
            'kv_transfer_config': '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}',
        }
        decoder_config = {
            'config_file': os.path.join(os.path.dirname(__file__), 'lmcache-decoder-config.yaml'),
            'port': 8200,
            'cuda_visible_devices': 1,
            'kv_transfer_config': '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}',
        }
        proxy_config = {
            'host': 'localhost',
            'port': 9000,
            'prefiller_host': 'localhost',
            'prefiller_port': 8100,
            'decoder_host': 'localhost',
            'decoder_port': 8200,
        }

        # Start all
        self.server_handles = start_vllm_disagg_server(
            checkpoints,
            prefiller_config,
            decoder_config,
            proxy_config
        )
        self.client = OpenAI(
            api_key="notset",
            base_url=f"http://localhost:{prefiller_config['port']}/v1"
        )
        self.model = self.client.models.list().data[0].id
        # Connect OpenAI client to proxy
        self.client = OpenAI(
            api_key="notset",
            base_url=f"http://{proxy_config['host']}:{proxy_config['port']}/v1"
        )

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