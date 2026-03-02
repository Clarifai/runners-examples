import io
import tarfile
from typing import Annotated, Any, Dict

import docker
from clarifai.runners.models.mcp_class import MCPModelClass
from fastmcp import FastMCP
from pydantic import Field

server = FastMCP(
    "python-execution-server",
    instructions="Execute Python code securely in Docker containers",
    stateless_http=True,
)

_docker_client = None


def get_docker_client():
    """
    Get or create Docker client.

    IMPORTANT: This function connects to your local Docker daemon.
    If the default connection fails, you MUST configure the Docker socket path
    for your specific system in the fallback options below.
    """
    global _docker_client
    if _docker_client is None:
        try:
            # Try default Docker environment connection
            _docker_client = docker.from_env()
            _docker_client.ping()
        except Exception as e:
            print(f"Default Docker connection failed: {e}")
            print("Trying alternative Docker socket paths...")

            # ============================================================================
            # CONFIGURATION REQUIRED: Update these paths for your local machine
            # ============================================================================
            # Common Docker socket paths:
            # - Docker Desktop (Mac): unix:///Users/YOUR_USERNAME/.docker/run/docker.sock
            # - Rancher Desktop (Mac): unix:///Users/YOUR_USERNAME/.rd/docker.sock
            # - Linux: unix:///var/run/docker.sock
            # - Docker Desktop (Windows): npipe:////./pipe/docker_engine
            # ============================================================================

            alternative_sockets = [
                # TODO: UPDATE THIS PATH - Replace with your actual Docker socket path
                'unix:///Users/YOUR_USERNAME/.rd/docker.sock',  # Example: Rancher Desktop on Mac

                # Standard paths (may work by default)
                'unix:///var/run/docker.sock',  # Standard Linux/Mac path
                'unix://~/.docker/run/docker.sock',  # Docker Desktop alternate path
            ]

            connected = False
            for socket_path in alternative_sockets:
                try:
                    print(f"Trying socket: {socket_path}")
                    _docker_client = docker.DockerClient(base_url=socket_path)
                    _docker_client.ping()
                    print(f"Successfully connected to Docker via: {socket_path}")
                    connected = True
                    break
                except Exception as socket_error:
                    print(f"Failed to connect via {socket_path}: {socket_error}")
                    continue

            if not connected:
                raise Exception(
                    f"Cannot connect to Docker daemon. Original error: {e}\n\n"
                    f"SETUP REQUIRED:\n"
                    f"1. Ensure Docker is running on your local machine\n"
                    f"2. Find your Docker socket path (run: docker context inspect)\n"
                    f"3. Update the 'alternative_sockets' list in model.py with your socket path\n"
                    f"4. Common paths:\n"
                    f"   - Rancher Desktop (Mac): unix:///Users/YOUR_USERNAME/.rd/docker.sock\n"
                    f"   - Docker Desktop (Mac): unix:///Users/YOUR_USERNAME/.docker/run/docker.sock\n"
                    f"   - Linux: unix:///var/run/docker.sock\n"
                )
    return _docker_client


def execute_python_code_fresh_container(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a fresh Docker container (OpenAI approach).
    Each execution gets a completely clean environment.
    """
    try:
        client = get_docker_client()

        # Pull Python image if not present
        try:
            client.images.get("python:3.11")
        except docker.errors.ImageNotFound:
            client.images.pull("python:3.11")

        # Create a temporary tar archive containing the script (like OpenAI)
        script_name = "script.py"
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            script_bytes = code.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=script_name)
            tarinfo.size = len(script_bytes)
            tar.addfile(tarinfo, io.BytesIO(script_bytes))
        tarstream.seek(0)

        # Start fresh container
        container = client.containers.create("python:3.11", command="sleep infinity", detach=True)

        try:
            container.start()
            # Put the script into the container
            container.put_archive(path="/tmp", data=tarstream.read())
            # Execute the script
            exec_result = container.exec_run(f"python /tmp/{script_name}")

            return {
                "stdout": exec_result.output.decode("utf-8", errors='replace'),
                "stderr": "",
                "status": exec_result.exit_code,
            }
        finally:
            # Always clean up container
            container.remove(force=True)

    except docker.errors.ContainerError as e:
        return {"stdout": "", "stderr": str(e), "status": 1}
    except Exception as e:
        return {"stdout": "", "stderr": f"Execution error: {str(e)}", "status": 1}


def execute_with_packages(code: str, packages: list = None) -> Dict[str, Any]:
    """
    Execute Python code with pre-installed packages.
    This is the key enhancement over #1 - allows package installation.
    """
    if packages:
        # Prepend package installation to the code
        install_code = "\n".join(
            [
                f"import subprocess; subprocess.run(['pip', 'install', '{pkg}'], check=True)"
                for pkg in packages
            ]
        )
        full_code = f"{install_code}\n\n{code}"
    else:
        full_code = code

    return execute_python_code_fresh_container(full_code)


@server.tool(
    "execute_with_packages", description="Execute Python code with packages pre-installed"
)
def execute_with_packages_tool(
    code: Annotated[str, Field(description="Python code to execute")],
    packages: Annotated[
        list[str], Field(description="List of packages to install before execution")
    ] = None,
) -> str:
    """
    Execute Python code with specified packages installed on top of the base Python image.
    This enables users to work with the full Python ecosystem.
    Example: execute_with_packages("import requests; print(requests.get('https://httpbin.org/json').json())", ["requests"])
    """
    if not code.strip():
        return "Error: No code provided"

    result = execute_with_packages(code, packages or [])

    if result["status"] == 0:
        output = "--- Execution Successful ---\n"
        if packages:
            output += f"Packages installed: {', '.join(packages)}\n\n"
        if result["stdout"].strip():
            output += result["stdout"]
        else:
            output += "(No output - use print() to see results)"
    else:
        output = f"--- Execution Error (status: {result['status']}) ---\n"
        if result["stderr"].strip():
            output += result["stderr"]
        if result["stdout"].strip():
            output += "\n--- Output ---\n" + result["stdout"]

    return output


class MyModel(MCPModelClass):
    def get_server(self) -> FastMCP:
        """Return the FastMCP server instance."""
        return server
