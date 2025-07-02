# Jina Embeddings v3 ONNX Model

This is a text embedding model based on **Jina AI v3**, optimized with **ONNX Runtime** for efficient inference. It generates high-quality embeddings suitable for **semantic search**, **text similarity**, and other **natural language processing (NLP)** tasks.

---

## ⚙️ Model Configuration

In the `config.yaml` file, **CPU memory** is set to **5Gi**, as the ONNX version of the model requires significant CPU resources.

---

## 🚀 Benefits of ONNX Implementation

- **Improved Performance**  
  Faster inference compared to the standard PyTorch implementation.

- **Cross-Platform Compatibility**  
  ONNX enables deployment across different hardware and platforms.

- **Reduced Memory Footprint**  
  Optimized for production environments.

- **Efficient Batch Processing**  
  Handles both single and batch text inputs effectively.

---

## 🧠 Model Overview

The model generates **dense vector embeddings** for input text using a state-of-the-art transformer architecture. Key components include:

- Token embedding layer  
- Multi-head attention mechanism  
- Mean pooling for sentence representation  
- L2 normalization of final embeddings

---

## 📁 Code Structure

The implementation consists of three main components:

1. **Model Loading (`load_model`)**
   - Initializes ONNX runtime session  
   - Loads tokenizer and configuration  
   - Sets up inference pipeline

2. **Text Processing (`tokenize_and_embed`)**
   - Tokenizes input text  
   - Prepares input for ONNX model  
   - Handles task-specific configurations

3. **Prediction (`predict`)**
   - Processes input text  
   - Applies mean pooling and normalization  
   - Returns normalized embedding vectors

---

## 💡 Usage (via Clarifai SDK)

```python
from clarifai.client.model import Model

# Initialize model
model = Model(
    model_url="model_url",
    pat="CLARIFAI_PAT"
)

# Get embeddings for a single text input
embeddings = model.predict(input="This is a sample text")
```
## 🧾 Example Output
The model returns a list of floating-point numbers representing the text embedding vector:

```python
[0.123, -0.456, 0.789, ...]  # 1024-dimensional vector
```

## 📌 Parameters
- input: str — The text string to be embedded

- returns: List[float] — The resulting normalized embedding vector

The model automatically handles:

- Input tokenization

- Attention masking

- Mean pooling

- L2 normalization

## 📚 More Information
For technical details and additional documentation, refer to the official [model card](https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/#parameter-dimensions).
