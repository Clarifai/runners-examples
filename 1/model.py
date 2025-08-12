from typing import List, Iterator
from threading import Thread
import os
import torch
import json

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.utils.openai_convertor import openai_response
from clarifai.runners.utils.data_utils import Param
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer)


class MyModel(OpenAIModelClass):
  """A custom runner for llama-3.2-1b-instruct llm that integrates with the Clarifai platform"""
  client = True
  model  = True
  def load_model(self):
    """Load the model here."""
    if torch.backends.mps.is_available():
        self.device = 'mps'
    elif torch.cuda.is_available():
        self.device = 'cuda'
    else:
        self.device = 'cpu'

    logger.info(f"Running on device: {self.device}")

    # Load checkpoints
    model_path = os.path.dirname(os.path.dirname(__file__))
    builder = ModelBuilder(model_path, download_validation_only=True)
    self.checkpoints = builder.config['checkpoints']['repo_id']
    logger.info(f"Loading model from: {self.checkpoints}")
    # Load model and tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoints,)
    self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token
    self.model = AutoModelForCausalLM.from_pretrained(
        self.checkpoints,
        low_cpu_mem_usage=True,
        device_map=self.device,
        torch_dtype=torch.bfloat16,
    )
    self.streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    self.chat_template = None
    logger.info("Done loading!")

  @OpenAIModelClass.method
  def predict(self,
              prompt: str ="",
              chat_history: List[dict] = None,
              max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
              temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
              top_p: float = Param(default=0.8, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", )) -> str:
    """
    Predict the response for the given prompt and chat history using the model.
    """
    # Construct chat-style messages
    messages = chat_history if chat_history else []
    if prompt:
        messages.append({
            "role": "user",
            "content":  prompt
        })
    
    inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]
    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "eos_token_id": self.tokenizer.eos_token_id,
    }

    output = self.model.generate(**inputs, **generation_kwargs)
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    num_tokens = len(generated_tokens)
    self.set_output_context(
        prompt_tokens=prompt_tokens,
        completion_tokens=num_tokens,
    )
    logger.info("token info: prompt_tokens=%d, completion_tokens=%d", prompt_tokens, num_tokens)
    return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

  @OpenAIModelClass.method
  def generate(self,
              prompt: str="",
              chat_history: List[dict] = None,
              max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
              temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
              top_p: float = Param(default=0.8, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", )) -> Iterator[str]:
      """Stream generated text tokens from a prompt + optional chat history."""

      # Construct chat-style messages
      messages = chat_history if chat_history else []
      if prompt:
          messages.append({
            "role": "user",
            "content":  prompt
        })
      # Track prompt tokens before generation
      prompt_tokens = len(self.tokenizer.apply_chat_template(
          messages, tokenize=True, add_generation_prompt=True
      ))
      
      generated_text = []
      
      response = self.chat(
          messages=messages,
          max_tokens=max_tokens,
          temperature=temperature,
          top_p=top_p
      )
      
      for each in response:
          if 'choices' in each and 'delta' in each['choices'][0] and 'content' in each['choices'][0]['delta']:
                  text_chunk = each['choices'][0]['delta']['content']
                  generated_text.append(text_chunk)
                  yield text_chunk
                  
      #calculate completion tokens
      completion_tokens = len(self.tokenizer.encode(''.join(generated_text), add_special_tokens=False))
      
      self.set_output_context(
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
      )
      logger.info("token info: prompt_tokens=%d, completion_tokens=%d", prompt_tokens, completion_tokens)
                  
  @OpenAIModelClass.method
  def chat(self,
          messages: List[dict],
          max_tokens: int = Param(default=512, description="The maximum number of tokens to generate. Shorter token lengths will provide faster performance.", ),
          temperature: float = Param(default=0.7, description="A decimal number that determines the degree of randomness in the response", ),
          top_p: float = Param(default=0.8, description="An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.", )
          ) -> Iterator[dict]:
      """
      Stream back JSON dicts for assistant messages.
      Example return format:
      {"role": "assistant", "content": [{"type": "text", "text": "response here"}]}
      """
      # Tokenize using chat template
      inputs = self.tokenizer.apply_chat_template(
          messages,
          tokenize=True,
          add_generation_prompt=True,
          return_tensors="pt"
      ).to(self.model.device)

      generation_kwargs = {
          "input_ids": inputs,
          "do_sample": True,
          "max_new_tokens": max_tokens,
          "temperature": temperature,
          "top_p": top_p,
          "eos_token_id": self.tokenizer.eos_token_id,
          "streamer": self.streamer
      }

      thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
      thread.start()

      # Accumulate response text
      for chunk in openai_response(self.streamer):
          yield chunk

      thread.join()
  
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
                "max_tokens": request_data.get('max_tokens', 1000)
            }
            prompt_tokens =  len(self.tokenizer.apply_chat_template(request_data['messages'], tokenize=True, add_generation_prompt=True, add_special_tokens=False))
            response = self.predict(chat_history=request_data['messages'],
                                    **options)
            if response :
                completion_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
                # wrap in choices.
                return json.dumps(
                  {
                    "choices": [
                      {
                        "message": response,
                        "finish_reason": "stop",
                      }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                  }
                )
            else:
                logger.error("No response received from the model.")
                return "No response received from the model."
        except Exception as e:
            return f"Error: {e}"

  @OpenAIModelClass.method
  def openai_stream_transport(self, msg: str) -> Iterator[str]:
    """Process an OpenAI-compatible request and stream the response.

    Args:
        msg: JSON string containing the request parameters including 'openai_endpoint'
      
    Returns:
        Iterator[str]: Streamed response chunks
    """
    try:
      request_data = json.loads(msg)
      
      options = {
          "temperature": request_data.get('temperature', 0.7),
          "top_p": request_data.get('top_p', 1.0),
          "max_tokens": request_data.get('max_tokens', 1000)
      }
      prompt_tokens = len(self.tokenizer.encode(str(request_data['messages']), add_special_tokens=False))
      response = self.generate(chat_history=request_data['messages'], **options)
      completion_text = []
      for chunk in response:
        completion_text.append(chunk)
        yield json.dumps(
          {
            "choices": [
              {
                "delta": {"content": chunk},
                "finish_reason": None,
              }
            ],
          }
        )
      
      completion_tokens = len(self.tokenizer.encode(''.join(completion_text), add_special_tokens=False))
      yield json.dumps(
        {
          "choices": [
            {
              "delta": {},
              "finish_reason": "stop",
            }
          ],
          "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
          }
        }
      )
      self.set_output_context(
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
      )
      logger.info("token info: prompt_tokens=%d, completion_tokens=%d", prompt_tokens, completion_tokens)
    except Exception as e:
      yield f"Error: {e}"
  
  def test(self):
    """Test the model here."""
    try:
      print("Testing predict...")
      # Test predict
      print(self.predict(prompt="What is the capital of India?",))
    except Exception as e:
      print("Error in predict", e)

    try:
      print("Testing generate...")
      # Test generate
      for each in self.generate(prompt="What is the capital of India?",):
        print(each, end="")
      print()
    except Exception as e:
      print("Error in generate", e)

    try:
      print("Testing chat...")
      messages = [
        {"role": "system", "content": "You are an helpful assistant."},
        {"role": "user", "content": "What is the capital of India?"},
      ]
      for each in self.chat(messages=messages,):
        print(each, end="")
      print()
    except Exception as e:
      print("Error in generate", e)
