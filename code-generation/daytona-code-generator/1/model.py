import os
import torch
from dotenv import load_dotenv
from typing import Dict

from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.utils.data_utils import Param

from transformers import AutoModelForCausalLM, AutoTokenizer
from daytona import Daytona

# Load environment variables from .env file
load_dotenv()

class MyModel(ModelClass):
    """A CPU-only model that uses Qwen2.5-0.5B to generate and execute Python code in a Daytona sandbox."""

    def load_model(self):
        """Load the model and initialize the Daytona client."""
        self.device = 'cpu'
        logger.info(f"Running on device: {self.device}")

        # Load LLM checkpoints
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        self.checkpoints = builder.download_checkpoints(stage="runtime")

        # Load LLM model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoints)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoints,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        # Initialize Daytona client
        # It will automatically look for DAYTONA_API_KEY in the environment

        try:
            with open(os.path.join(os.path.dirname(__file__), "daytona_api_key"), "r") as fh:
                os.environ["DAYTONA_API_KEY"] = fh.read().strip()
            self.daytona = Daytona()
            logger.info("Daytona client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Daytona client: {e}")
            self.daytona = None

        logger.info("Model and Daytona client loaded successfully!")

    def _generate_code(self, instruction: str) -> str:
        """Generates Python code from a natural language instruction."""
        # More specific prompt for the LLM
        prompt = f"""
As an expert Python programmer, your task is to write a concise, runnable script to accomplish the following task.
The script should be self-contained and not require any external files unless specified in the prompt.
Task: "{instruction}"
Please provide only the raw Python code, without any explanations, comments, or markdown formatting.
"""
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        generated_code = self.tokenizer.decode(response, skip_special_tokens=True)

        # Clean up the output to get only the code
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()

        return generated_code

    @ModelClass.method
    def predict(self, prompt: str = Param(default="", description="The user prompt to generate and execute Python code.")) -> Dict:
        """
        Generates Python code based on the prompt, executes it in a Daytona sandbox,
        and returns both the code and the execution result.
        """
        if not self.daytona:
            return {
                "error": "Daytona client is not initialized. Please check your API key.",
                "generated_code": None,
                "execution_result": None
            }

        if not prompt:
            return {
                "error": "Prompt cannot be empty.",
                "generated_code": None,
                "execution_result": None
            }

        logger.info(f"Generating code for prompt: {prompt}")
        generated_code = self._generate_code(prompt)
        logger.info(f"Generated code:\n{generated_code}")

        sandbox = None
        try:
            logger.info("Creating Daytona sandbox...")
            sandbox = self.daytona.create()
            logger.info(f"Sandbox created with ID: {sandbox.id}")

            logger.info("Executing code in sandbox...")
            execution = sandbox.process.code_run(generated_code)
            logger.info("Code execution finished.")

            if execution.exit_code == 0:
                result = execution.result
                error = None
            else:
                result = None
                error = execution.result # stderr is in result for code_run

            return {
                "generated_code": generated_code,
                "execution_result": result,
                "exit_code": execution.exit_code,
                "error": error
            }

        except Exception as e:
            logger.error(f"An error occurred during Daytona execution: {e}")
            return {
                "error": str(e),
                "generated_code": generated_code,
                "execution_result": None
            }
        finally:
            if sandbox:
                logger.info(f"Deleting sandbox {sandbox.id}...")
                sandbox.delete()
                logger.info("Sandbox deleted.")

    def test(self):
        """A simple test for the model."""
        logger.info("Running test...")
        test_prompt = "Create a pandas DataFrame with two columns, 'Name' and 'Age', and three rows of data. Then, print the DataFrame's shape."

        # This requires DAYTONA_API_KEY to be set in the environment
        if not os.getenv("DAYTONA_API_KEY"):
            logger.warning("DAYTONA_API_KEY not set. Skipping test.")
            return

        result = self.predict(prompt=test_prompt)

        print("--- Test Result ---")
        import json
        print(json.dumps(result, indent=2))
        print("--- End Test ---")

        assert "error" not in result or result["error"] is None
        assert result["generated_code"] is not None
        assert result["execution_result"] is not None
        assert result["exit_code"] == 0
        # The output of shape is a tuple, so its string representation should be present
        assert "(3, 2)" in result["execution_result"]
        logger.info("Test completed successfully.")

if __name__ == "__main__":
    # To run this test locally, you need to have a .env file with DAYTONA_API_KEY
    # and run `pip install -r ../requirements.txt`
    model = MyModel()
    model.load_model()
    model.test()
