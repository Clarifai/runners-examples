import base64
from typing import Dict, List

from clarifai.runners.utils.data_types import Image


def build_messages(prompt: str, image: Image, images: List[Image],
                   messages: List[Dict]) -> List[Dict]:
  """Construct OpenAI-compatible messages from input components."""
  openai_messages = []
  # Add previous conversation history
  if messages:
    openai_messages.extend(messages)

  content = []
  if prompt.strip():
    # Build content array for current message
    content.append({'type': 'text', 'text': prompt})
  # Add single image if present
  if image:
    content.append(_process_image(image))
  # Add multiple images if present
  if images:
    for img in images:
      content.append(_process_image(img))
      
  if content:
    # Append complete user message
    openai_messages.append({'role': 'user', 'content': content})

  return openai_messages


def _process_image(image: Image) -> Dict:
  """Convert Clarifai Image object to OpenAI image format."""
  if image.bytes:
    b64_img = base64.b64encode(image.bytes).decode('utf-8')
    return {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{b64_img}"}}
  elif image.url:
    return {'type': 'image_url', 'image_url': {'url': image.url}}
  else:
    raise ValueError("Image must contain either bytes or URL")


class OpenAIWrapper:

  def __init__(self, client: object, modalities: List[str] = None, model=None):
    self.client = client
    self.modalities = modalities or []
    self._validate_modalities()
    self.model_id = model
    

  def _validate_modalities(self):
    valid_modalities = {'image'}
    invalid = set(self.modalities) - valid_modalities
    if invalid:
      raise ValueError(f"Invalid modalities: {invalid}. Valid options: {valid_modalities}")

  def chat(self,
           prompt: str = "",
           image: Image = None,
           images: List[Image] = None,
           messages: List[Dict] = None,
           max_tokens: int = 512,
           temperature: float = 0.7,
           top_p: float = 0.8,
           stream=False) -> dict:
    """Process request through OpenAI API."""
    
    openai_messages = build_messages(prompt, image, images or [], messages or [])
    response = self.client.chat.completions.create(
        model=self.model_id,
        messages=openai_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream)

    return response