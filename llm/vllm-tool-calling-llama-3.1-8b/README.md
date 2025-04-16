### Model Prediction

Once the model is uploaded, you can easily make the prediction to the model using Clarifai SDK.

#### Prediction Method Structure

The client **exactly mirrors** the method signatures defined in your model's **model.py**:

| Model Implementation | Client Usage Pattern |
| --- | --- |
| **@ModelClass.method def func(self, prompt: str, image: Image = None, images: List[Image] = None, chat_history: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.func(prompt="Write 2000 word story")** |
| **@ModelClass.method def generate(self, prompt: str, image: Image = None, images: List[Image] = None, chat_history: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.generate(prompt="Write 2000 word story")** |
| **@ModelClass.method def chat(self, messages: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.chat(messages={'role': 'user', 'content': "Write 2000 word story", })** |

**Key Characteristics:**

* Method names match exactly what's defined in **model.py**
* Arguments/parameters preserve the same names and types
* Return types mirror the model's output definitions

#### Initializing the Model Client
First, instantiate your model with proper credentials:

```python
from clarifai.client.model import Model

# Initialize with explicit IDs
model = Model(
    user_id="model_user_id",
    app_id="model_app_id",
    model_id="model_id",
)

# Or initialize with model URL
model = Model(model_url="https://clarifai.com/model_user_id/model_app_id/models/model_id",)
```

#### Unary-Unary Prediction
> `chat_history` here must be same as Openai client `messages` object
```python
# Single input prediction
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                    "string",
                    "description":
                    "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "state", "unit"]
        }
    }
}]

prediction = model.predict(prompt="Can you tell me what the temperate will be in Dallas, in fahrenheit?", tools=tools)
import json
prediction = json.loads(prediction)
if prediction and type(prediction) is list and len(prediction) > 0 and prediction[0].get('function'):
    tool_calls_obj = prediction[0]['function']
    function_name  = tool_calls_obj[0]['function']['name']
    function_args = tool_calls_obj[0]['function']['arguments']
    print(f'function_name: {function_name}')
    print(f'function_args: {function_args}')
else:
    print('No function call found in the prediction response.')
    print(prediction)
```

#### Unary-Stream Prediction

#### Using `generate` Method
> `chat_history` here must be same as Openai client `messages` object
```python
image_url = "https://samples.clarifai.com/metro-north.jpg"
response_stream = model.generate(prompt= "what is in the image?", image = Image.from_url(image_url), temperature=0.4, max_tokens=100)

for text_chunk in response_stream:
    print(text_chunk, end="", flush=True)
```

#### Using `chat` Method
> `messages` here must be same as Openai client `messages` object
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                    "string",
                    "description":
                    "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "state", "unit"]
        }
    }
}]

messages = [{
    "role": "user",
    "content": "Hi! How are you doing today?"
}, {
    "role": "assistant",
    "content": "I'm doing well! How can I help you?"
}, {
    "role": "user",
    "content": "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
}]

stream_response = model.chat(messages=messages, tools = tools, max_tokens=150, temperature=1, top_p=0.8)


chunks = []
for chunk in stream_response:
    chunks.append(chunk)
    if chunk['choices'][0]['delta'] and chunk['choices'][0]['delta'].get('tool_calls'):
        print(chunk['choices'][0]['delta']['tool_calls'][0])
    else:
        print(chunk['choices'][0]['delta'])

arguments = []
tool_call_idx = -1
for chunk in chunks:

    if chunk['choices'][0]['delta'].get('tool_calls'):
        tool_call = chunk['choices'][0]['delta']['tool_calls'][0]

        if tool_call['index'] != tool_call_idx:
            if tool_call_idx >= 0:
                print(
                    f"streamed tool call arguments: {arguments[tool_call_idx]}"
                )
            tool_call_idx = chunk['choices'][0]['delta']['tool_calls'][0]['index']
            arguments.append("")
        if tool_call.get('id'):
            print(f"streamed tool call id: {tool_call['id']} ")

        if tool_call.get('function'):
            if tool_call['function'].get('name'):
                print(f"streamed tool call name: {tool_call['function']['name']}")

            if tool_call['function'].get('arguments'):
                arguments[tool_call_idx] += tool_call['function']['arguments']

if len(arguments):
    print(f"streamed tool call arguments: {arguments[-1]}")

```
