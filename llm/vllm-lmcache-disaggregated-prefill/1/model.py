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

def launch_vllm_server(
    role: str,
    checkpoints: str,
    gpu: str,
    port: int,
    lmcache_config_file: str,
    kv_rpc_port: str,
):
    """
    Launch a vLLM server for either the Prefiller or Decoder.
    """
    from clarifai.runners.utils.model_utils import execute_shell_command
    kv_transfer_config = (
        '{'
        f'"kv_connector":"LMCacheConnectorV1",'
        f'"kv_role":"{role}",'
        '"kv_connector_extra_config":{'
        '"discard_partial_chunks": false, '
        f'"lmcache_rpc_port": "{kv_rpc_port}"'
        "}}"
    )
    
    cmds = [
        f"UCX_TLS=cuda_ipc,cuda_copy,tcp",
        f"LMCACHE_CONFIG_FILE={lmcache_config_file}",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        PYTHON_EXEC, "-m", "vllm.entrypoints.openai.api_server",
        checkpoints,
        "--port", str(port),
        "--disable-log-requests",
        "--kv-transfer-config", kv_transfer_config
    ]

    cmd = " ".join(cmds)
    logger.info(f"Launching vLLM {role} server: {cmd}")
    return execute_shell_command(cmd)

def launch_proxy_server(host: str, port: int, prefiller_port: int, decoder_port: int):
    """
    Launch the disaggregated proxy server that coordinates between prefiller and decoder.
    """
    from clarifai.runners.utils.model_utils import execute_shell_command

    proxy_script = os.path.join(
        os.path.dirname(__file__), "disagg_proxy_server.py"
    )

    cmd = f"{PYTHON_EXEC} {proxy_script} --host {host} --port {port} --prefiller-host {host} --prefiller-port {prefiller_port} --decoder-host {host} --decoder-port {decoder_port}"
    logger.info(f"Launching Disaggregated Proxy Server: {cmd}")
    return execute_shell_command(cmd)

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

      # Config
      prefiller_port = 8100
      decoder_port = 8200
      proxy_port = 9000
      host = "localhost"

      # Launch prefiller on GPU 0
      self.prefiller_proc = launch_vllm_server(
          role="kv_producer",
          checkpoints=checkpoints,
          gpu="0",
          port=prefiller_port,
          lmcache_config_file=os.path.join(
              os.path.dirname(__file__), "lmcache-prefiller-config.yaml"
          ),
          kv_rpc_port="producer1",
      )

      # Launch decoder on GPU 1
      self.decoder_proc = launch_vllm_server(
          role="kv_consumer",
          checkpoints=checkpoints,
          gpu="1",
          port=decoder_port,
          lmcache_config_file=os.path.join(
              os.path.dirname(__file__), "lmcache-decoder-config.yaml"
          ),
          kv_rpc_port="consumer1",
      )

      # Launch proxy server
      self.proxy_proc = launch_proxy_server(
          host=host, port=proxy_port, prefiller_port=prefiller_port, decoder_port=decoder_port
      )

      # Wait for proxy to be ready
      from clarifai.runners.utils.model_utils import wait_for_server

      wait_for_server(f"http://{host}:{proxy_port}")
      logger.info(f"Disaggregated Prefill Proxy is up at http://{host}:{proxy_port}")

      # Set up OpenAI client to point to proxy
      self.client = OpenAI(api_key="notset", base_url=f"http://{host}:{proxy_port}/v1")
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