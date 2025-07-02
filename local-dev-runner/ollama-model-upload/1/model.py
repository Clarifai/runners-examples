from typing import List, Iterator
import os

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger

from ollama import Client

def run_ollama_server(model_name: str = 'llama3.2'):
    """
    start the Ollama server.
    """
    from clarifai.runners.utils.model_utils import execute_shell_command, terminate_process
    
    os.environ['OLLAMA_HOST'] = '127.0.0.1:2333'
    
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
        self.model_name = 'llama3.2-vision:latest'
        
        #start ollama server
        run_ollama_server(model_name=self.model_name)
        
        self.model = Client(host=os.environ['OLLAMA_HOST'])
        logger.info(f"Ollama model loaded successfully: {self.model}")
        
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
