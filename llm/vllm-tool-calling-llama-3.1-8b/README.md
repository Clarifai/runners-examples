### Model Prediction

Once the model is uploaded, you can easily make the prediction to the model using Clarifai SDK.

#### Prediction Method Structure

The client **exactly mirrors** the method signatures defined in your model's **model.py**:

| Model Implementation | Client Usage Pattern |
| --- | --- |
| **@ModelClass.method def predict(self, prompt: str, chat_history: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.predict(prompt="Write 2000 word story")** |
| **@ModelClass.method def generate(self, prompt: str, chat_history: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.generate(prompt="Write 2000 word story")** |

**Key Characteristics:**

* Method names match exactly what's defined in **model.py**
* Arguments/parameters preserve the same names and types
* Return types mirror the model's output definitions

---

## Quickstart

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


---

### 1\. Install the Clarifai Python SDK

```bash
pip install clarifai
```

### 2\. Set your Clarifai Personal Access Token (PAT)

Retrieve your PAT from your Clarifai account security settings.

```bash
export CLARIFAI_PAT="your_personal_access_token"
```

---

## OpenAI-Compatible API Usage

GPT-4.1 is accessible via the OpenAI-compatible API endpoint. You can utilize the OpenAI Python package as follows:

```python
from openai import OpenAI

client = OpenAI(
    api_key="CLARIFAI_PAT",  # Your Clarifai PAT key
    base_url="https://api.clarifai.com/v2/ext/openai/v1"  # Clarifai's OpenAI-compatible API endpoint
)

response = client.chat.completions.create(
    model="https://clarifai.com/model_user_id/model_app_id/models/model_id",  # Clarifai model URL
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain the concept of quantum entanglement?"}
    ],
    tools=None,
    tool_choice=None,
    max_completion_tokens=100,
    temperature=0.7,
    stream=True,
)
```

---

## Predict with Text

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/model_user_id/model_app_id/models/model_id")
prompt = "What are the key differences between classical and quantum computing?"
result = model.predict(prompt)
print("Predict response:", result)
```

---

## Streaming/Generate Example

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/model_user_id/model_app_id/models/model_id")
print("Generate response:")
for chunk in model.generate(prompt="Discuss the implications of AI in modern healthcare."):
    print(chunk, end='', flush=True)
```

---

## Tool Calling Example

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/model_user_id/model_app_id/models/model_id")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieve the current stock price for a given company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["ticker"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

prediction = model.predict(
    prompt="What is the current stock price of Tesla (TSLA)?",
    tools=tools,
    tool_choice='auto',
    max_tokens=1024,
    temperature=0.5,
)

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