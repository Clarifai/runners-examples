# Jina Embeddings v3 ONNX Model
This is a text embedding model that uses the Jina AI v3 model optimized with ONNX runtime for efficient inference. The model generates high-quality text embeddings suitable for semantic search, text similarity, and other NLP tasks.

### Model Config

In the `config.yaml` file CPU memory is set as 5Gis, since the onnx version of model is used and it requires significant CPU memory.

### Benefits of ONNX Implementation


**Improved Performance**: ONNX runtime optimization provides faster inference compared to standard PyTorch implementation

**Cross-Platform Compatibility**: ONNX enables deployment across different hardware and platforms

**Reduced Memory Footprint**: Optimized model architecture for production environments

**Efficient Batch Processing**: Handles both single and batch text inputs effectively: 
ONNX runtime optimization provides faster inference compared to standard PyTorch implementation

**Cross-Platform Compatibility**: ONNX enables deployment across different hardware and platforms

**Reduced Memory Footprint**: Optimized model architecture for production environments

**Efficient Batch Processing**: Handles both single and batch text inputs effectively.

### Model Overview
The model provides dense vector embeddings (embeddings) for input text using state-of-the-art transformer architecture. Key components:

- Token embedding layer
- Multi-head attention mechanism
- Mean pooling for sentence representation
- L2 normalization of final embeddings

### Code Structure
The implementation consists of three main components:

1. Model Loading (load_model):
    - Initializes ONNX runtime session
    - Loads tokenizer and model configuration
    - Sets up the inference pipeline
2. Text Processing (tokenize_and_embed):
    - Tokenizes input text
    - Prepares inputs for ONNX model
    - Handles task-specific configurations
3. Prediction (predict):
    - Processes input text
    - Applies mean pooling and normalization
    - Returns normalized embedding vectors

### Usage
Using Clarifai SDK
```python
from clarifai.client.model import Model

# Initialize model
model = Model(
    model_url="model_url",
    pat="CALRIFAI_PAT"
)

# Get embeddings for single text
embeddings = model.predict(input="This is a sample text")

```

### Example Output
The model returns a list of floating-point numbers representing the text embedding vector:
```
[0.123, -0.456, 0.789, ...]  # 1024-dimensional vector
```
### Parameters
- input: Text string to be embedded
- Returns: List[float] representing the embedding vector


The model automatically handles:
- Input text tokenization
- Attention masking
- Mean pooling
- L2 normalization
For technical details and additional documentation, see the [model card](https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/#parameter-dimensions).
