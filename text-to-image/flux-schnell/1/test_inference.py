# Use this script to test uploaded model locally

import requests
from PIL import Image
from io import BytesIO
from clarifai.client.model import Model

model = Model(
    "https://clarifai.com/phatvo/text-generation-pythonic/models/test-flux", deployment_id="dasfasd")
image = model.predict(
    prompt=["A Ghibli animated orange cat, panicked about a deadline, sits in front of a Banana-brand laptop."], negative_prompt="Ugly, cute")

# Do something else with the output
image.to_pil().save("output_image.jpg")

