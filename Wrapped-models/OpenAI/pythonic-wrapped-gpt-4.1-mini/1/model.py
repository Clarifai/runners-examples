import os
import sys

sys.path.append(os.path.dirname(__file__))
from openai_client_wrapper import OpenAIWrapper
from typing import List

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Image, Stream
from openai import OpenAI


# Set your OpenAI API key here
OPENAI_API_KEY = 'your_openai_api_key_here'

class GPT4_1_Mini(ModelClass):
    """
    A custom runner that integrates with the Clarifai platform and uses Server inference
    to process inputs, including text and images.
    """
    
    def load_model(self):
        """Load the model here and start the server."""
        # Create client
        self.client = OpenAIWrapper(
        client=OpenAI(api_key=OPENAI_API_KEY,base_url="https://api.openai.com/v1/"),
        modalities=["image"],
        model = "gpt-4.1-mini",
        )
        
        # log that system is ready
        print("OpenAI gpt-4.1-mini model loaded successfully!")
        
    
    @ModelClass.method
    def predict(self,
              prompt: str,
              image: Image = None,
              images: List[Image] = None,
              chat_history: List[dict] = None,
              max_tokens: int = 512,
              temperature: float = 0.7,
              top_p: float = 0.8) -> str:
        """This is the method that will be called when the runner is run. It takes in an input and
        returns an output.
        """
        return self.client.chat(
            prompt=prompt,
            image=image,
            images=images,
            messages=chat_history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p).choices[0].message.content
        
    
    @ModelClass.method
    def generate(self,
                prompt: str,
                image: Image = None,
                images: List[Image] = None,
                chat_history: List[dict] = None,
                max_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.8) -> Stream[str]:
        """Example yielding a whole batch of streamed stuff back."""
        for chunk in self.client.chat(
            prompt=prompt,
            image=image,
            images=images,
            messages=chat_history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True):
          if chunk.choices:
            text = (chunk.choices[0].delta.content
                    if (chunk and chunk.choices[0].delta.content) is not None else '')
            yield text
            
    @ModelClass.method
    def chat(self,
           messages: List[dict],
           max_tokens: int = 512,
           temperature: float = 0.7,
           top_p: float = 0.8) -> Stream[dict]:
        """Chat with the model."""
        for chunk in self.client.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True):
          yield chunk.to_dict()
          
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

        
            
            