# Stable Diffusion 2 Depth Model
A powerful image generation model that utilizes depth information to create more coherent and realistic images based on text prompts and reference images.

## Overview
This implementation integrates the stabilityai/stable-diffusion-2-depth model with Clarifai's platform, enabling depth-aware image generation and modification. The model is particularly effective for:

- Creating variations of images while preserving spatial relationships
- Generating new images with accurate depth perception
- Modifying existing images while maintaining structural coherence

### Usage
```python
from clarifai.client.model import Model
from clarifai.runners.utils.data_types import Image

model_url="https://clarifai.com/user_id/app_id/models/stable-diffusion-2-depth"


model = Model(url=model_url)

img=Image(url="http://images.cocodataset.org/val2017/000000039769.jpg")

# Model Predict//
model_prediction = model.predict(prompt="two tigers",
                                 image=img,
                                 negative_prompt="bad, deformed, uglybad anatomy"
                                 )

print(model_prediction) #returns a bytes representation of image.
```
### Display image
```python
from PIL import Image as PILimage
import io

# Convert bytes to PIL Image
def bytes_to_pil_image(image_bytes):
    return PILimage.open(io.BytesIO(image_bytes))

img = bytes_to_pil_image(model_prediction.bytes)
img.show()
```

### Parameters

- prompt (str): Text description of the desired image
- image (Image, optional): Reference image for guided generation
- negative_prompt (str, optional): Text description of unwanted elements
- strength (float, default=0.8): Control parameter for image modification intensity
- mask (Image, optional): Mask for selective modification
