from typing import List

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Image, Stream

from vllm import LLM, SamplingParams


def qwen2_vl_template(question: str, modality: str, images: list):
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        
    if modality == 'image':
        placeholders = [{"type": "image", "image": data} 
                        for img in images 
                        if (data := img.url or img.bytes)]
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role":
            "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": question
                },
            ],
        }]
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        prompt = processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        
        image_data, _ = process_vision_info(messages)
        return prompt, image_data

def apply_prompt_template(text,
                   images = None):
    if images:
        prompt_template, img_bytes = qwen2_vl_template(text, 'image', images)
        return prompt_template, img_bytes
    else:
        prompt_template = text
        return prompt_template
        
def chat_completion(llm,
                     prompt,
                     images,
                     temperature,
                     max_tokens,
                     top_p
    ):
    
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    if images:
        prompt, img_bytes = apply_prompt_template(prompt,images)
        outputs = llm.generate({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [img_bytes]
                        },
                    },
                use_tqdm=False,
                sampling_params=params
                )
    else:
        outputs = llm.generate({
                "prompt": prompt,
                },
                use_tqdm=False,
                sampling_params=params
                )
    for o in outputs:
        generated_text = o.outputs[0].text
    return generated_text
        
class MyRunner(ModelClass):
  """
  A custom runner that integrates with the Clarifai platform and uses Server inference
  to process inputs, including text and images.
  """

  def load_model(self):
    """Load the model here and start the  server."""
    
    self.client = LLM(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=16000,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        trust_remote_code=True
    )
    
    return self.client

  @ModelClass.method
  def predict(self,
              prompt: str,
              images: List[Image]=None,
              chat_history: List[dict]=None,
              max_tokens: int = 512,
              temperature: float = 0.7,
              top_p: float = 0.8) -> str :
    
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    
    return chat_completion(llm=self.client,
                prompt=prompt,
                images=images,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
    )
  
  @ModelClass.method
  def generate(self,
              prompt: str,
              images: List[Image]=None,
              chat_history: List[dict]=None,
              max_tokens: int = 512,
              temperature: float = 0.7,
              top_p: float = 0.8) -> str:
  
    return chat_completion(llm=self.client,
                prompt=prompt,
                images=images,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
    )
    
  
  @ModelClass.method 
  def chat(self,
           messages: List[dict],
           max_tokens: int = 512,
           temperature: float = 0.7,
           top_p: float = 0.8) -> Stream[dict]:
        
    raise NotImplementedError("Chat method is not implemented for the models.")