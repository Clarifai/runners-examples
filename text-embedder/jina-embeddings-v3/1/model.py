import os
import sys
from typing import List

# Clarifai Runner utilities:
# - ModelBuilder: handles checkpoint resolution & download logic
# - OpenAIModelClass: base class for OpenAI-compatible runners
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger

# Official OpenAI Python client.
# vLLM exposes an OpenAI-compatible API, so we can reuse this client.
from openai import OpenAI

# Absolute path to the Python interpreter running this process.
# Used to ensure the vLLM server is launched with the same Python environment.
PYTHON_EXEC = sys.executable


# ----------------------------------------------------------------------
# vLLM OpenAI-compatible server launcher
# ----------------------------------------------------------------------
def vllm_openai_server(checkpoints, **kwargs) -> object:
    """
    Launch a vLLM OpenAI-compatible API server as a subprocess.

    Args:
        checkpoints (str):
            Path or HuggingFace repo ID of the model to load.
        **kwargs:
            Extra vLLM server CLI arguments (e.g. dtype, port, host).

    Returns:
        server (object):
            A lightweight object containing server metadata and process handle.
    """

    # Utility functions provided by Clarifai for:
    # - spawning shell commands
    # - waiting until HTTP server is ready
    # - terminating subprocesses cleanly
    from clarifai.runners.utils.model_utils import (
        execute_shell_command,
        wait_for_server,
        terminate_process,
    )

    # Base command to start vLLM's OpenAI-compatible API server
    cmds = [
        PYTHON_EXEC,                                # Use current Python binary
        "-m",
        "vllm.entrypoints.openai.api_server",       # vLLM OpenAI server entrypoint
        "--model",
        checkpoints,                                # Model checkpoint path or repo ID
    ]

    # Convert keyword arguments into CLI flags
    # Example:
    #   dtype="float16"  -> --dtype float16
    #   trust_remote_code=True -> --trust-remote-code
    for key, value in kwargs.items():
        if value is None:
            continue
        param_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmds.append(f"--{param_name}")
        else:
            cmds.extend([f"--{param_name}", str(value)])

    # Create a lightweight "server" object dynamically
    # This avoids defining a full class just for metadata storage
    server = type(
        "Server",
        (),
        {
            "host": kwargs.get("host", "0.0.0.0"),
            "port": kwargs.get("port", 23333),
            "backend": "vllm",
            "process": None,
        },
    )()

    try:
        # Launch the vLLM server process
        server.process = execute_shell_command(" ".join(cmds))

        # Wait until the server is ready to accept requests
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info(f"vLLM server running at http://{server.host}:{server.port}")
    except Exception as e:
        # Cleanup if server failed to start
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start vLLM server: {e}")

    return server


# ----------------------------------------------------------------------
# Embeddings Runner
# ----------------------------------------------------------------------
class VLLMJinaEmbeddingsModel(OpenAIModelClass):
    """
    Clarifai Runner for `jinaai/jina-embeddings-v3`.

    - Uses vLLM as the inference backend
    - Exposes an OpenAI-compatible `/v1/embeddings` endpoint
    - Integrates seamlessly with Clarifai's OpenAIModelClass
    """

    # Required flags for OpenAIModelClass
    client = True
    model = True

    def load_model(self):
        """
        Called once when the runner starts.

        Responsibilities:
        - Resolve and download model checkpoints
        - Launch vLLM OpenAI-compatible server
        - Initialize OpenAI client pointing to vLLM
        """

        # vLLM server configuration
        server_args = {
            "dtype": "float16",                 # Use float16 for faster inference
            "trust_remote_code": True,          # Trust custom model code from HuggingFace
            "gpu_memory_utilization": 0.9,      # Max GPU memory usage
            "tensor_parallel_size": 1,          # Single-GPU setup
            "port": 23333,
            "host": "localhost",
        }

        # Model directory containing Clarifai config.yaml
        model_path = os.path.dirname(os.path.dirname(__file__))

        # ModelBuilder handles:
        # - reading model config
        # - checkpoint resolution
        # - conditional downloading
        builder = ModelBuilder(model_path, download_validation_only=True)

        # Stage indicates when checkpoints should be fetched
        # Possible values: "build", "runtime", etc.
        stage = builder.config["checkpoints"]["when"]

        # Default checkpoint reference (usually HuggingFace repo ID)
        checkpoints = builder.config["checkpoints"]["repo_id"]

        if stage in ["build", "runtime"]:
            checkpoints = builder.download_checkpoints(stage=stage)

        # Start vLLM OpenAI-compatible server
        self.server = vllm_openai_server(checkpoints, **server_args)

        # OpenAI client
        self.client = OpenAI(
            api_key="notset",
            base_url=f"http://{self.server.host}:{self.server.port}/v1",
        )

        # Retrieve the model ID exposed by vLLM
        # vLLM dynamically registers models at runtime
        self.model = self.client.models.list().data[0].id
        logger.info(f"Loaded embedding model: {self.model}")

    @OpenAIModelClass.method
    def predict(
        self,
        texts: List[str],
        ) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        """
        # Call the OpenAI-compatible embeddings endpoint
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # If token usage information is available,
        # report it back to Clarifai for metering & observability
        if resp.usage:
            self.set_output_context(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=1, # embeddings endpoint has no completion tokens, 1 added for consistency
            )
        # Extract and return embeddings
        return [d.embedding for d in resp.data]
