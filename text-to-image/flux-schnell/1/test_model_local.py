import os
import sys

sys.path.append(os.path.dirname(__file__))

from model import TextToImageModel

model = TextToImageModel()
model.load_model()

image = model.predict(
    prompt="A Ghibli animated orange cat, panicked about a deadline, sits in front of a Banana-brand laptop.", 
    negative_prompt="Ugly, cute", guidance_scale=7)
print(image)
image.to_pil().save("tmp/flux_schnell.jpg")

images = model.create(
    prompt=["A Ghibli animated orange cat, panicked about a deadline, sits in front of a Banana-brand laptop."]*3, 
    negative_prompt=["Ugly, cute"]*2, guidance_scale=7)