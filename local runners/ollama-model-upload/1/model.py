from typing import Any, Dict, List, Iterator
import os
import json

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages

from openai import OpenAI

# Set default host
if not os.environ.get('OLLAMA_HOST'):
  os.environ["OLLAMA_HOST"] = '127.0.0.1:23333'
OLLAMA_HOST = os.environ.get('OLLAMA_HOST')


def run_ollama_server(model_name: str = 'llama3.2'):
    """
    start the Ollama server.
    """
    from clarifai.runners.utils.model_utils import execute_shell_command, terminate_process
    

    try:
        logger.info(f"Starting Ollama server in the host: {OLLAMA_HOST}")
        start_process = execute_shell_command("ollama serve")
        if start_process:
            pull_model=execute_shell_command(f"ollama pull {model_name}")
            logger.info(f"Model {model_name} pulled successfully.")
            logger.info(f"Ollama server started successfully on {OLLAMA_HOST}")

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
        #set the model name here or via OLLAMA_MODEL_NAME
        self.model_name = os.environ.get("OLLAMA_MODEL_NAME", 'granite3.3:2b')#'devstral:latest')
        
        #start ollama server
        run_ollama_server(model_name=self.model_name)

        self.client = OpenAI(
                api_key="notset",
                base_url= f"http://{OLLAMA_HOST}/v1")
        self.model = self.client.models.list().data[0].id
        logger.info(f"Ollama model loaded successfully: {self.model}")
        
  
    @OpenAIModelClass.method
    def predict(self,
                prompt: str,
                chat_history: List[dict] = None,
                tools: List[dict] = None,
                tool_choice: str = None,
                max_tokens: int = Param(default=2048, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass."), 
                ) -> str:
      """
      This method is used to predict the response for the given prompt and chat history using the model and tools.
      """

      if tools is not None and tool_choice is None:
          tool_choice = "auto"
              
      messages = build_openai_messages(prompt=prompt, messages=chat_history)
      response = self.client.chat.completions.create(
          model=self.model,
          messages=messages,
          tools=tools,
          tool_choice=tool_choice,
          max_completion_tokens=max_tokens,
          temperature=temperature,
          top_p=top_p)
        
      if response.choices[0] and response.choices[0].message.tool_calls:
        # If the response contains tool calls, return as a string
        tool_calls = response.choices[0].message.tool_calls
        tool_calls_json = json.dumps([tc.to_dict() for tc in tool_calls], indent=2)
        return tool_calls_json
      else:
        # Otherwise, return the content of the first choice
        return response.choices[0].message.content
      

    @OpenAIModelClass.method
    def generate(self,
                prompt: str,
                chat_history: List[dict] = None,
                tools: List[dict] = None,
                tool_choice: str = None,
                max_tokens: int = Param(default=2048, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
                temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
                top_p: float = Param(default=0.95, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.")) -> Iterator[str]:
      """
      This method is used to stream generated text tokens from a prompt + optional chat history and tools.
      """
      messages = build_openai_messages(prompt=prompt, messages=chat_history)
      response = self.client.chat.completions.create(
          model=self.model,
          messages=messages,
          tools=tools,
          tool_choice=tool_choice,
          max_completion_tokens=max_tokens,
          temperature=temperature,
          top_p=top_p,
          stream=True)
      
      for chunk in response:
        if chunk.choices:
          if chunk.choices[0].delta.tool_calls:
            # If the response contains tool calls, return the first one as a string
            tool_calls = chunk.choices[0].delta.tool_calls
            tool_calls_json = [tc.to_dict() for tc in tool_calls]
            # Convert to JSON string
            json_string = json.dumps(tool_calls_json, indent=2)
            # Yield the JSON string
            yield json_string
          else:
            # Otherwise, return the content of the first choice
            text = (chunk.choices[0].delta.content
                    if (chunk and chunk.choices[0].delta.content) is not None else '')
            yield text
