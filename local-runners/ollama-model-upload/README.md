![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)

# Running Ollama models using Clarifai local-runner 

A Python integration for running Large Language Models with Ollama in your local device (mac) and inference from anywhere using clarifai's local-runner functionality.

This Model integration is specifically run and tested on mac. 

## 🔍 Prerequisites
---
- [Ollama](https://ollama.com/download) should be installed and running in your local device.

> **Note for Windows users:** After installing Ollama, you need to restart your machine to ensure the updated environment variables take effect.

## 🚀 Running model locally

1. **Install [Clarifai](https://github.com/Clarifai/clarifai-python)** : 
```bash
pip install clarifai #>=11.5.5
```
Install [OpenAI](https://github.com/openai/openai-python) (optional) - This enables us to call the model in openAI compatible way.
```bash
pip install openai
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

Quickstart tips on ollama models and use cases

**Multimodal** - `llama3.2-vision:latest`
**Tool calling** - `llama3-groq-tool-use:latest`
**Coding agent** - `devstral:latest`


## **Run the model using clarifai local-runner**:
Run any ollama model in your local machine using local-runner with just 2 lines of code.
- Refer to this [link](https://docs.clarifai.com/compute/models/upload/run-locally) for more information on how to use local-runners  and it's use cases.

1. **Initialize model from ollama**
   ```bash
   clarifai model init --toolkit ollama
   ```
   
   Options
   
   You can call any model from [Ollama library](https://ollama.com/library) with following options
   
   `--model-name` - **Name of your ollama model (defaults to llama3.2)**
   
   `--port` - **Port where the model is running (defaults to 23333)**
   
   `--context-length` - **Set the context-length for model (defaults to 8192)**

   For example: serving a gemma3n model with context of 16k at port 8008
   ```bash
   clarifai model init --toolkit --ollama --port 8008 --context-length 16000
   ```
   
3. **Load and run the model locally**
   ```bash
   clarifai model local-runner .
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
                model="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model-2",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
print(f"Response: {response.choices[0].message.content}")

```
### Inference with Clarifai SDK Predict
`model_url` can be taken from your account, where the model instance is created. 

Model URL will follow below format - `https://clarifai.com/user-id/app-id/models/model-id` .

You can also get the `app-id` ,`user-id` ,`model-id` from `ollama-model-upload/config.yaml` .

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model-2")

prompt = "Hello, Good morning!"
result = model.predict(prompt)

print("Predict response:", result)
```

### Inference with Clarifai SDK Generate
```python
from clarifai.client.model import Model

# Initialize model
model_url="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model"
model = Model(url=model_url)

# Generate text
prompt = "Hello, Good morning!"
response = model.generate(prompt)

for chunk in response:
    print(chunk, end='')
```

### Multimodal inference with Clarifai SDK

```python
from clarifai.client.model import Model
from clarifai.runners.utils.data_types import Image

image_url = "https://samples.clarifai.com/metro-north.jpg"
image_obj = Image(url=image_url)

# Initialize model
model_url="https://clarifai.com/user-id/local-dev-runner-app/models/local-dev-model"

model = Model(url=model_url)

#Predict
#response = model.predict(messages=message)

#Generate
response = model.predict(
    prompt="Describe this image.",
    image=image_obj,
    max_tokens=1024,
    temperature=0.5,
)

for chunk in response:
    print(chunk, end='')
```

## Customizing ollama model
### 📁 Model file Structure
---
Below is the file structure of how the model file would look like once you have initialized the model. For further customization of model templates or optimizations you can edit the `model.py` file inside the folder `1/` and run the model.
```
ollama-model-upload/
   ├── 1/
   │   └── model.py          # Main model implementation
   │    
   ├── config.yaml           # Model configuration
   └── requirements.txt      # Project dependencies

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
