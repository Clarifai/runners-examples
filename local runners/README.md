![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)

# Running Ollama models using Clarifai local-dev runners

A Python integration for running Large Language Models with Ollama in your local device (mac) and inference from anywhere using clarifai's local-dev runner functionality.

This Model integration is specifically run and tested on mac. 


## 📁 Project Structure
---

```
ollama-model-upload/
   ├── 1/
   │   └── model.py          # Main model implementation
   │    
   ├── config.yaml           # Model configuration
   └── requirements.txt      # Project dependencies

```
## 🔍 Prerequisites
---
- [Ollama](https://ollama.com/download) should be installed and running in your local device.

> **Note for Windows users:** After installing Ollama, you need to restart your machine to ensure the updated environment variables take effect.

## 🚀 Running model locally

1. **Install [Clarifai](https://github.com/Clarifai/clarifai-python)** : 
```bash
pip install clarifai #>=11.5.5
```

2. **Configure context in clarifai config**:
- Login to clarifai.
- Create context with user-id and personal access token.
- For more info on setting clarifai context - Refer to this [documentation](https://docs.clarifai.com/compute/models/upload/run-locally/#log-in)
```bash
clarifai login
```
For further info run this in terminal
```bash
clarifai config --help
```

3. **Setting model name, context length and host**:
 - [Ollama](https://ollama.com/search) provides with wide variety of models which you can run in your local system. Be cautious on the model size when you run it locally. Always monitor the resource usage.

 - To set model name in the script and run any model. Change the model name in this script `ollama-model-upload/1/model.py` in the `load_model` method under `OllamaModelClass` .

 ```python
 #Set model name
 self.model_name = "llama3.2-vision:latest"
 ```
 - You can modify the model context length by setting the below variable. Default it will be using 4096.
 ```python
 os.environ['OLLAMA_CONTEXT_LENGTH'] = '8192' #or 16000 etc.
 ```
 - Additionally you can also change the host and port for your ollama server to run by modifying the `"OLLAMA_HOST"` value in `run_ollama_server` function.
 ```python
 os.environ['OLLAMA_HOST'] = '127.0.0.1:2333'
 ```

4. **Run the model using clarifai local-dev**:
- Once you have made all the changes, run the below command to start an local-dev runner in your system. 
- Refer to this [link](https://docs.clarifai.com/compute/models/upload/run-locally) for more information on how to use local-dev runners and it's use cases.
- Model path should point to the model files. 
```bash
clarifai model local-dev /ollama-model-upload
```

## 💻 Usage Example
The runner will be started in your local machine and now it will be ready for inference.

local runner models can be also inferenced using OpenAI compatible endpoint function.
### Set Clarifai PAT
Refer this [guide](https://docs.clarifai.com/control/authentication/pat/#how-to-create-a-pat-on-the-platform) on how to obtain one from the platform.
```python
os.environ["CLARIFAI_PAT"] = "YOUR_CLARIFAI_PAT"
```
### Inference using OpenAI compatible method
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ['CLARIFAI_PAT'],  
)
# Replace with your user-id
response = client.chat.completions.create(
    model="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model",
    messages=[
        {"role": "system", "content": "Talk like a pirate."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
    temperature=0.7,
    stream=False, # stream=True also works, just iterator over the response
)
print(response)

# For printing stream response.
#for chunk in response:
    #print(chunk.choices[0].message['content'], end='')
```
### Multimodal inference using OpenAI compatible method for ollama models

Since ollama handles images in different way, we can directly pass the bytes representation of image in the message dictionary.
```python
from pathlib import Path
import base64

# local path to image
#path = "local/path/of/image.png"
#image_base64 = base64.b64encode(Path(path).read_bytes()).decode()

# Or download image from URL and pass it as bytes 
def get_image_base64(image_url):
    """Download image and convert to base64."""
    response = requests.get(image_url)
    return base64.b64encode(response.content).decode('utf-8')

image_url = "https://samples.clarifai.com/cat1.jpeg"
image_base64 = get_image_base64(image_url)

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ['CLARIFAI_PAT'],  
)

# Replace with your user-id
response = client.chat.completions.create(
            model="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model",
            messages=[
                {
                    "role": "user",
                    'content': 'what is in this image?',
                    'images': [image_base64],
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
print(f"Response: {response.choices[0].message.content}")

```
### Inference with Clarifai SDK Predict
`model_url` can be taken from your account, where the model instance is created. 

Model URL will follow below format - `https://clarifai.com/user-id/app-id/models/model-id` .

You can also get the `app-id` ,`user-id` ,`model-id` from `ollama-model-upload/config.yaml` .

```python
from clarifai.client.model import Model

# Initialize model
model_url="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model"

model = Model(url=model_url)

# Generate text
messages = [{"role": "user", "content": "Why the sky is blue"}]
response = model.predict(messages)

print(response)
```

### Inference with Clarifai SDK Generate
```python
from clarifai.client.model import Model

# Initialize model
model_url="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model"

model = Model(url=model_url)

# Generate text
messages = [{"role": "user", "content": "Why the sky is blue"}]
response = model.generate(messages, options={'temperature':0.7})

for chunk in response:
    print(chunk, end='')
```

### Multimodal inference with Clarifai SDK

```python
from pathlib import Path
import base64
from clarifai.client.model import Model

# Initialize model
model_url="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model"

model = Model(url=model_url)

#path to the image
path = "local/path/of/image.png"

#Convert image into base64 str
img = base64.b64encode(Path(path).read_bytes()).decode()

message=[
    {
      'role': 'user',
      'content': 'what is in this image?',
      'images': [img],
    }
  ]

#Predict
#response = model.predict(messages=message)

#Generate
response = model.generate(messages=message)

for chunk in response:
    print(chunk, end='')
```

For more references on how to call the ollama model, refer to this [example](https://github.com/ollama/ollama-python/tree/main/examples) repository.
## 📦 Features

- Automated Ollama server management
- Model pulling and loading
- Support for text generation and chat completions
- Stream response support
- Error handling and logging

## 🤝 Reference
 - [Clarifai-python](https://github.com/Clarifai/clarifai-python)
 - [Clarifai Examples](https://github.com/Clarifai/examples/blob/main/README.md)
 - [Clarifai Docs](https://docs.clarifai.com/compute/models/upload/run-locally/#use-cases-for-local-dev-runners)
 - [Ollama-python](https://github.com/ollama/ollama-python)
 - [Ollama](https://ollama.com/)

