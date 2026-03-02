# Multimodal Model Examples

This directory contains examples for deploying models that process multiple input modalities (vision + language, OCR, etc.).

## Categories

### vision-language/
Models that understand both images and text.

**Examples:**
- **qwen2_5-vl-3b-instruct-vllm** - Vision-language model for image understanding and chat
  - 3B parameters, 20Gi GPU
  - Supports image + text input
  - Chat-style interactions about images

**Use cases:**
- Image captioning and description
- Visual question answering
- Document understanding
- Scene analysis

### ocr/
Optical Character Recognition models for extracting text from images.

**Examples:**
- **deepseek-ocr-sglang** - DeepSeek OCR using SGLang framework
  - Efficient OCR processing
  - Vision-language capabilities
  - 20Gi GPU requirement

- **nanonets-ocr-s** - Nanonets OCR model
  - Custom OCR implementation
  - Production-ready
  - 20Gi GPU requirement

**Use cases:**
- Document digitization
- Receipt/invoice processing
- Form extraction
- Text extraction from images

## Framework Support

### vLLM
- Best for vision-language models requiring high throughput
- Efficient memory management with PagedAttention
- Supports streaming for interactive applications

### SGLang
- Optimized for structured output generation
- Efficient prompt caching for repetitive OCR tasks
- Fast inference for vision tasks

### Custom Implementations
- Specialized models with unique requirements
- Direct framework integration
- Fine-tuned preprocessing pipelines

## Input Format

Multimodal models typically accept:

```python
{
    "inputs": [
        {
            "data": {
                "image": {"url": "https://..."},
                "text": {"raw": "What's in this image?"}
            }
        }
    ]
}
```

Or for local testing:

```python
from clarifai_grpc.grpc.api import resources_pb2

input_proto = resources_pb2.Input(
    data=resources_pb2.Data(
        image=resources_pb2.Image(url="https://..."),
        text=resources_pb2.Text(raw="Describe this image")
    )
)
```

## Configuration Tips

### GPU Memory Requirements
- **20Gi**: Standard for most multimodal models
- Vision models typically need more memory than text-only models
- Consider batch size based on available memory

### Model Classes
- **OpenAIModelClass**: For chat-style vision-language models
- **ModelClass**: For custom OCR and specialized implementations

## Choosing the Right Model

**For visual question answering:**
- → `vision-language/qwen2_5-vl-3b-instruct-vllm`

**For document OCR:**
- → `ocr/deepseek-ocr-sglang` (efficient, structured output)
- → `ocr/nanonets-ocr-s` (production-ready, specialized)

**For general image understanding:**
- → `vision-language/qwen2_5-vl-3b-instruct-vllm`

## Deployment Workflow

1. **Choose your model** based on task requirements
2. **Review example README** for specific configuration
3. **Prepare test images** for validation
4. **Customize config.yaml** with compute settings
5. **Deploy**: `clarifai model upload`
6. **Test with image inputs** via API or web interface

## Performance Considerations

- **Image preprocessing**: Most models resize/normalize automatically
- **Batch processing**: Process multiple images efficiently
- **Streaming**: Vision-language models support streaming responses
- **Input resolution**: Higher resolution = more GPU memory
- **Context length**: Vision tokens count toward context limit

## Common Use Cases

### Vision-Language Models
- **Product catalog description**: Generate descriptions from product images
- **Visual assistance**: Answer questions about images
- **Content moderation**: Analyze image content with context
- **Accessibility**: Generate alt-text for images

### OCR Models
- **Invoice processing**: Extract structured data from receipts
- **Document digitization**: Convert scanned documents to text
- **Form automation**: Extract fields from forms
- **License plate recognition**: Read text from vehicle images

## API Examples

### Vision-Language Chat

```python
from clarifai.client.model import Model

model = Model(
    user_id="your-user-id",
    app_id="your-app-id",
    model_id="your-model-id"
)

response = model.predict(
    inputs=[{
        "data": {
            "image": {"url": "https://example.com/image.jpg"},
            "text": {"raw": "What objects are in this image?"}
        }
    }]
)

print(response.outputs[0].data.text.raw)
```

### OCR Extraction

```python
response = model.predict(
    inputs=[{
        "data": {
            "image": {"url": "https://example.com/document.jpg"}
        }
    }]
)

extracted_text = response.outputs[0].data.text.raw
print(f"Extracted: {extracted_text}")
```

## Additional Resources

- [Getting Started](../00-getting-started/) - Learn the basics
- [LLM Examples](../llm/) - Text-only language models
- [Framework Index](../FRAMEWORK_INDEX.md) - Browse by framework
- [Clarifai Docs](https://docs.clarifai.com/) - Platform documentation

## Contributing

When adding multimodal examples:
1. Place in appropriate subdirectory (vision-language/ or ocr/)
2. Include README with frontmatter metadata
3. Specify input format clearly
4. Document image preprocessing requirements
5. Include example images in documentation
