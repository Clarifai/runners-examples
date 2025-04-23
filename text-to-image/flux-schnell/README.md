![FLUX.1 [schnell] Grid](./schnell_grid.jpeg)

`FLUX.1 [schnell]` is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.
For more information, please read [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/).

# Key Features
1. Cutting-edge output quality and competitive prompt following, matching the performance of closed source alternatives.
2. Trained using latent adversarial diffusion distillation, `FLUX.1 [schnell]` can generate high-quality images in only 1 to 4 steps.
3. Released under the `apache-2.0` licence, the model can be used for personal, scientific, and commercial purposes.

**Checkpoint**: [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

# Usage

## Sample code

```python
from clarifai.client.model import Model
from clarifai.runners.utils.data_types import Image

inference_params = dict(
  num_inference_steps = 50, # A higher value improves image quality but increases processing time. Recommended range: 50–100.
  eta=1
)

prompt=["A Ghibli animated orange cat, panicked about a deadline, sits in front of a Banana-brand laptop."]*3, 

# Model Predict
model = Model("https://clarifai.com/black-forest-labs/models/flux_1-schnell")
images: List[Image] = model.predict(prompt=prompt, **inference_params)
images[0].to_pil().save("tmp.jpg")
```

### Parameters
* prompt (`List[str]`): The prompt or prompts to guide the image generation.
* prompt_2 (`List[str]`, *optional*): The prompt to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead.
* negative_prompt (`List[str]`, *optional*): The prompt to guide what to not include in image generation. Ignored when not using guidance (guidance_scale < 1).
* negative_prompt_2 (`List[str]`, *optional*): The negative_prompt to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is will be used instead.
* height (`int`, *optional*, defaults to model.unet.config.sample_size * model.vae_scale_factor): The height in pixels of the generated image. This is set to 1024 by default for the best results.
* width (`int`, *optional*, defaults to model.unet.config.sample_size * model.vae_scale_factor): The width in pixels of the generated image. This is set to 1024 by default for the best results.
* num_inference_steps (`int`, *optional*, defaults to 28): The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
* guidance_scale (`float`, *optional*, defaults to 3.5):
    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
    `guidance_scale` is defined as `w` of equation 2. of [Imagen
    Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
    1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
    usually at the expense of lower image quality.
* seed (`int`, *optional*, defaults to None):
    Seed value passed to `torch.Generator("cpu").manual_seed(seed)` (see more [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)) to make generation deterministic.
* sigmas (`List[float]`, *optional*): Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.
* max_sequence_length (`int` defaults to `256`): Maximum sequence length to use with the prompt.


---
# Limitations
- This model is not intended or able to provide factual information.
- As a statistical model this checkpoint might amplify existing societal biases.
- The model may fail to generate output that matches the prompts.
- Prompt following is heavily influenced by the prompting-style.

# Out-of-Scope Use
The model and its derivatives may not be used

- In any way that violates any applicable national, federal, state, local or international law or regulation.
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way; including but not limited to the solicitation, creation, acquisition, or dissemination of child exploitative content.
- To generate or disseminate verifiably false information and/or content with the purpose of harming others.
- To generate or disseminate personal identifiable information that can be used to harm an individual.
- To harass, abuse, threaten, stalk, or bully individuals or groups of individuals.
- To create non-consensual nudity or illegal pornographic content.
- For fully automated decision making that adversely impacts an individual's legal rights or otherwise creates or modifies a binding, enforceable obligation.
- Generating or facilitating large-scale disinformation campaigns.