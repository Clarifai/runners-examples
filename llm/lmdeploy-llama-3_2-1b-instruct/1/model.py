import os
from typing import List

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Stream
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from transformers import AutoTokenizer


HEADER_LOG = "<|start_header_id|>assistant<|end_header_id|>"


class MyRunner(ModelClass):
    """
    A custom runner that integrates with the Clarifai platform and uses lmdeploy inference
    to process inputs, including text and images.
    """

    def load_model(self):
        """Load the model here."""
        os.path.join(os.path.dirname(__file__))
        # Load checkpoints
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        checkpoints = builder.download_checkpoints(stage="runtime")
        backend_config = TurbomindEngineConfig(tp=1)
        self.pipe = pipeline(checkpoints, backend_config=backend_config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoints)

    @ModelClass.method
    def predict(self,
                prompt: str,
                max_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9) -> str:
        """This is the method that will be called when the runner is run. It takes in an input and
        returns an output.
        """
        messages = [{"role": "user", "content": prompt}]
        gen_config = GenerationConfig(
            temperature=temperature, max_new_tokens=max_tokens, top_p=top_p)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        res = self.pipe(prompt, gen_config=gen_config)
        text = res.text.replace(HEADER_LOG, "")
        return text.replace("\n\n", "")

    @ModelClass.method
    def generate(self,
                 prompt: str,
                 max_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> Stream[str]:
        """Example yielding a whole batch of streamed stuff back."""
        messages = [{"role": "user", "content": prompt}]
        gen_config = GenerationConfig(
            temperature=temperature, max_new_tokens=max_tokens, top_p=top_p)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        for item in self.pipe.stream_infer(prompt, gen_config=gen_config):
            text = item.text.replace(HEADER_LOG, "")
            yield text.replace("\n\n", "")

    @ModelClass.method
    def chat(self,
             messages: List[dict],
             max_tokens: int = 256,
             temperature: float = 0.7,
             top_p: float = 0.9) -> Stream[str]:
        """Chat with the model."""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        gen_config = GenerationConfig(
            temperature=temperature, max_new_tokens=max_tokens, top_p=top_p)
        for item in self.pipe.stream_infer(prompt, gen_config=gen_config):
            text = item.text.replace(HEADER_LOG, "")
            yield text.replace("\n\n", "")

    def test(self):
        """Test the model here."""
        try:
            print("Testing predict...")
            print(self.predict(prompt="Hello, how are you?"))
        except Exception as e:
            print("Error in predict", e)

        try:
            print("Testing generate...")
            for each in self.generate(prompt="Hello, how are you?"):
                print(each, end="")
        except Exception as e:
            print("Error in generate", e)

        try:
            print("\n-----------------------------------------------\n")
            print("Testing chat...")
            messages = [{"role": "user", "content": "Hello, how are you?"}]
            for each in self.chat(messages=messages):
                print(each, end="")
        except Exception as e:
            print("Error in chat", e)


if __name__ == "__main__":
    # This is for local testing
    runner = MyRunner()
    runner.load_model()
    runner.test()
