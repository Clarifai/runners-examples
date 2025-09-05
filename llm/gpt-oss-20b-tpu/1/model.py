import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_types import Image
from clarifai.runners.utils.data_utils import Param
from clarifai.utils.logging import logger
import torch_xla.core.xla_model as xm
from typing import List, Iterator


class GptOss20bTpu(OpenAIModelClass):
    tokenizer = None
    model = None

    def load_model(self):
        """
        Load the model and tokenizer from Hugging Face and move the model to the TPU device.
        """
        self.model_name = "openai/gpt-oss-20b"
        logger.info(f"Loading model {self.model_name}...")
        try:
            # Get the TPU device
            self.device = xm.xla_device()
            logger.info(f"Using device: {self.device}")

            # Load the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Move the model to the TPU device
            self.model.to(self.device)
            logger.info(f"Model {self.model_name} loaded successfully on TPU.")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")


    def _create_generation_config(self, max_tokens, temperature, top_p):
        return {
            "max_length": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str,
        max_tokens: int = Param(default=1024, description="The maximum number of tokens to generate."),
        temperature: float = Param(default=0.7, description="The temperature for sampling."),
        top_p: float = Param(default=0.95, description="The top-p value for sampling."),
    ) -> str:
        """
        Generate text from a prompt.
        """
        logger.info("Starting prediction...")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generation_config = self._create_generation_config(max_tokens, temperature, top_p)
            outputs = self.model.generate(**inputs, **generation_config)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Prediction completed successfully.")
            return generated_text

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Failed to generate text: {e}")

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str,
        max_tokens: int = Param(default=1024, description="The maximum number of tokens to generate."),
        temperature: float = Param(default=0.7, description="The temperature for sampling."),
        top_p: float = Param(default=0.95, description="The top-p value for sampling."),
    ) -> Iterator[str]:
        """
        Generate text from a prompt and stream the output.
        """
        logger.info("Starting generation...")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generation_config = self._create_generation_config(max_tokens, temperature, top_p)

            from transformers import TextStreamer
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

            # Run generation in a separate thread for streaming
            import threading
            generation_kwargs = dict(inputs, **generation_config, streamer=streamer)
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                yield new_text

            thread.join()
            logger.info("Generation completed successfully.")

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise RuntimeError(f"Failed to generate text: {e}")
