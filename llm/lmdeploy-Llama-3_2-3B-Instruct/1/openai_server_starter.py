import os
import subprocess
import sys
import threading
import time
from typing import List

from clarifai.utils.logging import logger
import psutil
import signal

PYTHON_EXEC = sys.executable


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


class OpenAI_APIServer:

  def __init__(self, **kwargs):
    self.server_started_event = threading.Event()
    self.process = None
    self.backend = None
    self.server_thread = None

  def __del__(self, *exc):
    # This is important
    # close the server when exit the program
    logger.info("Killing the server.")
    self.close()

  def close(self):
    if self.process:
      logger.info(f"Kill process: {self.process.pid}")
      try:
        kill_process_tree(self.process.pid)
      except:
        self.process.kill()
        self.process.terminate()

  def wait_for_startup(self):
    self.server_started_event.wait()

  def validate_if_server_start(self, line: str):
    line_lower = line.lower()
    if self.backend in ["vllm", "sglang", "lmdeploy"]:
      if self.backend == "vllm":
        return "application startup complete" in line_lower or "vllm api server on" in line_lower
      else:
        return f" running on http://{self.host}:" in line.strip()
    elif self.backend == "llamacpp":
      return "waiting for new tasks" in line_lower
    elif self.backend == "tgi":
      return "Connected" in line.strip()

  def _start_server(self, cmds):
    try:
      env = os.environ.copy()
      env["VLLM_USAGE_SOURCE"] = "production-docker-image"
      self.process = subprocess.Popen(
          cmds,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          encoding="utf-8",
          errors="ignore"
      )
      for line in self.process.stdout:
        logger.info("Server Log:  " + line.strip())
        if self.validate_if_server_start(line):
          self.server_started_event.set()
          # break
    except Exception as e:
      if self.process:
        self.process.terminate()
      raise RuntimeError(f"Failed to start Server server: {e}")

  def start_server_thread(self, cmds: str):
    try:
      # Start the  server in a separate thread
      self.server_thread = threading.Thread(
          target=self._start_server, args=(cmds,), daemon=None)
      self.server_thread.start()

      # Wait for the server to start
      self.wait_for_startup()
    except Exception as e:
      raise Exception(e)

  @classmethod
  def from_lmdeploy_backend(
      cls,
      checkpoints: str,
      backend: str = "turbomind",
      cache_max_entry_count=0.9,
      tensor_parallel_size=1,
      max_prefill_token_num=4096,
      dtype='auto',
      quantization_format: str = None,
      quant_policy: int = 0,
      chat_template: str = None,
      max_batch_size=16,
      device="cuda",
      server_name="0.0.0.0",
      server_port=23333,
      additional_list_args: List[str] = []
  ):
    """Run lmdeploy OpenAI compatible server

    Args:
        checkpoints (str): model id or path
        backend (str, optional): turbomind or pytorch. Defaults to "turbomind".
        cache_max_entry_count (float, optional): reserved mem for cache. Defaults to 0.9.
        tensor_parallel_size (int, optional): n gpus. Defaults to 1.
        max_prefill_token_num (int, optional): prefill token, the higher the more GPU mems are used. Defaults to 4096.
        dtype (str, optional): dtype. Defaults to 'auto'.
        quantization_format (str, optional): quantization {awq, gptq}. Defaults to None.
        quant_policy (int, optional): KV cache quant policty {0, 4, 8} bits, 0 means not using quantization. Defaults to 0.
        chat_template (str, optional): Chat template. To see all chatempltes, run `lmdeploy list`. Defaults to None.
        max_batch_size (int, optional): batch size. Defaults to 16.
        device (str, optional): device. Defaults to "cuda".
        server_port (int, optional): port. Defaults to 23333.
        server_name (str, optional): host name. Defaults to "0.0.0.0".
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at [github](https://github.com/InternLM/lmdeploy/blob/e8c8e7a019eb67430d7eeea74295813a6de0a780/lmdeploy/cli/serve.py#L83). Defaults to [].

    """
    # lmdeploy serve api_server $MODEL_DIR --backend $LMDEPLOY_BE --server-port 23333
    cmds = [
        PYTHON_EXEC,
        '-m',
        'lmdeploy',
        'serve',
        'api_server',
        checkpoints,
        '--dtype',
        str(dtype),
        '--backend',
        str(backend),
        '--tp',
        str(tensor_parallel_size),
        '--server-port',
        str(server_port),
        '--server-name',
        str(server_name),
        '--cache-max-entry-count',
        str(cache_max_entry_count),
        '--quant-policy',
        str(quant_policy),
        '--device',
        str(device),
    ]

    if quantization_format:
      cmds += ['--model-format', str(quantization_format)]

    if chat_template:
      cmds += ['--chat-template', str(chat_template)]

    if max_batch_size:
      cmds += ['--max-batch-size', str(max_batch_size)]

    if max_prefill_token_num:
      cmds += ['--max-prefill-token-num', str(max_prefill_token_num)]

    cmds += additional_list_args
    logger.info("CMDS to run `lmdeploy` server: " + " ".join(cmds))

    _self = cls()

    _self.host = server_name
    _self.port = server_port
    _self.backend = "lmdeploy"
    _self.start_server_thread(cmds)

    return _self
