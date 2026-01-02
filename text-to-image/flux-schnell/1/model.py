import os
from typing import List

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.data_types import Image
from clarifai.runners.models.model_builder import ModelBuilder

from diffusers import FluxPipeline
import torch


class TextToImageModel(ModelClass):
  """
  A custom runner for the FLUX model that integrates with the Clarifai platform.
  """

  def load_model(self):
    """Load the model here."""
    # "black-forest-labs/FLUX.1-schnell"
    
    self.device = "cuda"
    
    model_path = os.path.dirname(os.path.dirname(__file__))
    builder = ModelBuilder(model_path, download_validation_only=True)
    checkpoints = builder.download_checkpoints(stage="runtime")

    # load model and scheduler
    self.pipeline = FluxPipeline.from_pretrained(
      checkpoints,
      torch_dtype=torch.bfloat16
    )
    
    self.pipeline = self.pipeline.to(self.device)

  @ModelClass.method
  def predict(
    self,
    prompt: str,
    num_inference_steps: int = Param(default=28, description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."),
    guidance_scale: float = Param(default=3.5, description="The `guidance_scale` controls how strongly the model follows the conditioning input during generation."),
    negative_prompt: str = Param(default="", description="The prompt to guide what to not include in image generation. Ignored when not using guidance (guidance_scale < 1)"),
    true_cfg_scale: float = Param(default=1.0, description="When > 1.0 and a provided negative_prompt, enables true classifier-free guidance"),
    height: int = Param(default=1024, description="The height in pixels of the generated image. This is set to 1024 by default for the best results."),
    width: int = Param(default=1024, description="The width in pixels of the generated image. This is set to 1024 by default for the best results."),
    max_sequence_length: int = Param(default=256, description="Maximum sequence length to use with the prompt"),
    seed: int = Param(default=None, description="Seed value to make generation deterministic."),
    # No need
    sigmas: List[float] = None,
  ) -> Image:
    
    image = self.pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        width=width,
        height=height,
        true_cfg_scale=true_cfg_scale,
        generator=torch.Generator("cpu").manual_seed(seed) if seed else None,
        sigmas=sigmas,
    ).images[0]
    
    # this is important, delete all model cache to avoid OOM
    torch.cuda.empty_cache()
    
    return Image.from_pil(image)
  
  @ModelClass.method
  def generate_images(
      self,
      prompt: str,
      num_inference_steps: int = Param(
          default=50, description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."),
      num_images_per_prompt: int = Param(
          default=1, description="The number of images to generate per prompt."),
      guidance_scale: float = Param(
          default=None, description="The `guidance_scale` controls how strongly the model follows the conditioning input during generation."),
      negative_prompt: str = Param(
          default="", description="The prompt to guide what to not include in image generation. Ignored when not using guidance (guidance_scale < 1)"),
      true_cfg_scale: float = Param(
          default=4.0, description="When > 1.0 and a provided negative_prompt, enables true classifier-free guidance"),
      height: int = Param(
          default=1024, description="The height in pixels of the generated image. This is set to 1024 by default for the best results."),
      width: int = Param(
          default=1024, description="The width in pixels of the generated image. This is set to 1024 by default for the best results."),
      seed: int = Param(
          default=None, description="Seed value to make generation deterministic."),
  ) -> Image:

    images = self.pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        true_cfg_scale=true_cfg_scale,
        generator=torch.Generator("cpu").manual_seed(seed) if seed else None,
    ).images

    # this is important, delete all model cache to avoid OOM
    torch.cuda.empty_cache()

    return [Image.from_pil(image) for image in images]


  def test(self):
    """ 
    Test cases only executed when running `clarifai model test-locally`
    """
    image = self.predict(
    prompt="A Ghibli animated orange cat, panicked about a deadline, sits in front of a Banana-brand laptop.", 
    negative_prompt="Ugly, cute", guidance_scale=7)
    print(image)

    images = self.create(
        prompt=["A Ghibli animated orange cat, panicked about a deadline, sits in front of a Banana-brand laptop."]*3, 
        negative_prompt=["Ugly, cute"]*2, guidance_scale=7)
    print(images)