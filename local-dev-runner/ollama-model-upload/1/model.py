from typing import List, Iterator
import os
import json

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger

from ollama import Client

def run_ollama_server(model_name: str = 'llama3.2'):
    """
    start the Ollama server.
    """
    from clarifai.runners.utils.model_utils import execute_shell_command, terminate_process

    os.environ['OLLAMA_HOST'] = '127.0.0.1:2333'
    os.environ['OLLAMA_CONTEXT_LENGTH'] = '131072'
    cmd = f"ollama serve"
    try:
        logger.info(f"Starting Ollama server with command: {cmd}")
        start_process = execute_shell_command(cmd)
        if start_process:
            pull_model=execute_shell_command(f"ollama pull {model_name}")
            logger.info(f"Model {model_name} pulled successfully.")
            logger.info(f"Ollama server started successfully on {os.environ['OLLAMA_HOST']}")

    except Exception as e:
        logger.error(f"Error starting Ollama server: {e}")
        if 'start_process' in locals():
            terminate_process(start_process)
        raise RuntimeError(f"Failed to start Ollama server: {e}")


class OllamaModelClass(OpenAIModelClass):

    client =  True
    model = True

    def load_model(self):
        """
        Load the Ollama model.
        """
        #set the model name here
        # self.model_name = 'llama3.2-vision:latest'
        self.model_name = 'devstral:latest'

        #start ollama server
        run_ollama_server(model_name=self.model_name)

        self.model = Client(host=os.environ['OLLAMA_HOST'])
        logger.info(f"Ollama model loaded successfully: {self.model}")

    def _set_usage(self, resp):
        """Set the usage information from the response."""
        if resp.prompt_eval_count:
          self.set_output_context(
            prompt_tokens=resp.prompt_eval_count,
            completion_tokens=resp.eval_count,
          )


    @OpenAIModelClass.method
    def openai_transport(self, msg: str) -> str:
        """Process an OpenAI-compatible request and send it to the appropriate OpenAI endpoint.

        Args:
            msg: JSON string containing the request parameters including 'openai_endpoint'

        Returns:
            JSON string containing the response or error
        """
        try:
            request_data = json.loads(msg)

            options = {
                "temperature": request_data.get('temperature', 0.7),
                "top_p": request_data.get('top_p', 1.0),
                "max_tokens": request_data.get('max_tokens', 1000),
                "stop": request_data.get('stop', None)
            }

            response = self.model.chat(model=self.model_name,
                                       messages=request_data['messages'],
                                       options=options)

            if response and response.get('message'):
                self._set_usage(response)

                d = response.model_dump()
                print(d)
                # wrap in choices.
                return json.dumps(
                  {
                    "choices": [
                      {
                        "message": d['message'],
                        "finish_reason": d.get('done_reason', None)
                      }
                    ]
                  }
                )
            else:
                logger.error("No response received from the model.")
                return "No response received from the model."
        except Exception as e:
            return f"Error: {e}"



    @OpenAIModelClass.method
    def openai_stream_transport(self, msg: str) -> Iterator[str]:
        """Process an OpenAI-compatible request and return a streaming response iterator.
        This method is used when stream=True and returns an iterator of strings directly,
        without converting to a list or JSON serializing. Supports chat completions and responses endpoints.

        Args:
            msg: The request as a JSON string.

        Returns:
            Iterator[str]: An iterator yielding text chunks from the streaming response.
        """
        try:
            request_data = json.loads(msg)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)
            if endpoint not in [self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES]:
                raise ValueError("Streaming is only supported for chat completions and responses.")


            if endpoint == self.ENDPOINT_RESPONSES:
                raise Exception("responses format not supported yet")
                # Handle responses endpoint
                stream_response = self._route_request(endpoint, request_data)
                for chunk in stream_response:
                    yield json.dumps(chunk.model_dump())
            else:


                options = {
                    "temperature": request_data.get('temperature', 0.7),
                    "top_p": request_data.get('top_p', 1.0),
                    "max_tokens": request_data.get('max_tokens', 1000),
                    "stop": request_data.get('stop', None)
                }

                response = self.model.chat(model=self.model_name,
                                           messages=request_data['messages'],
                                           options=options,
                                           stream=True)

                for chunk in response:
                    self._set_usage(chunk)
                    d = chunk.model_dump()
                    print(d)
                    yield json.dumps({"choices": [{"message": d['message'], "finish_reason": d.get('done_reason', None)}]})

        except Exception as e:
            yield f"Error: {e}"


    @OpenAIModelClass.method
    def predict(self, messages: List[dict],
                format: str = None,
                options: dict = {}) -> str:
        """
        Generate text based on the provided prompt.
        Args:
            prompt (str): The input text prompt.
            chat_history (List[dict]): The chat history for context.
        Returns:
            str: The generated text response.
        """

        response = self.model.chat(model=self.model_name,
                                   messages=messages,
                                   format=format,
                                   options=options)

        if response and response.get('message'):
            return response['message']['content']
        else:
            logger.error("No response received from the model.")
            return "No response received from the model."

    @OpenAIModelClass.method
    def generate(
        self,
        messages: List[dict],
        format: str = None,
        options: dict = {}
    ) -> Iterator[str]:
        """
        Generate text based on the provided prompt.
        Args:
            messages (List[dict]): The input messages for the chat.
            format (str, optional): The format of the response.
            options (dict, optional): Additional inference params like (temperature) for the model.

        Returns:
            Iterator[str]: An iterator yielding generated text chunks.
        """
        response = self.model.chat(model=self.model_name,
                                   messages=messages,
                                   format=format,
                                    options=options,
                                   stream=True)

        for chunk in response:
            if chunk and chunk.get('message'):
                yield chunk['message']['content']
            else:
                logger.error("No response received from the model.")
                yield "No response received from the model."
