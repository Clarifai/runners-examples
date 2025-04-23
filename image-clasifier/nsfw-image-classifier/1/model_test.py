
import os
import sys

sys.path.append(os.path.dirname(__file__))
import requests
from model import ImageClassifierModel
from io import BytesIO
from clarifai.runners.utils.data_types import Image, Video
model = ImageClassifierModel()
model.load_model()



ims = Image(bytes=requests.get("https://imagez.tmz.com/image/07/o/2020/06/23/07104855c7a6436697163947c6752aa1_lg.jpg").content)
concepts = model.predict(image=ims)
print(concepts)


vid = Video(bytes=requests.get("https://samples.clarifai.com/beer.mp4").content)
generate = model.generate(video=vid)
for each in generate:
    print(each)