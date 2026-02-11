# Large Language Model (LLM) Examples

This directory contains examples for deploying Large Language Models on Clarifai using various frameworks optimized for inference.

## Available Examples

This directory includes 9 LLM examples ranging from lightweight models (135M parameters) to large agentic workflows (20B+ parameters):

- **Simple text generation** - Basic chat and completion models
- **Optimized inference** - High-throughput serving with vLLM, SGLang, LMDeploy
- **Tool calling** - Models that can execute functions
- **Agentic workflows** - Multi-step reasoning with MCP server integration
- **Embeddings** - Text embedding generation

## Framework Comparison

| Framework | Strengths | Best For | GPU Efficiency |
|-----------|-----------|----------|----------------|
| **vLLM** | High throughput, PagedAttention, continuous batching | Production serving, high traffic | ⭐⭐⭐⭐⭐ |
| **SGLang** | Fast structured generation, efficient prompt caching | Structured outputs, repetitive prompts | ⭐⭐⭐⭐⭐ |
| **LMDeploy** | Optimized for specific architectures, low latency | Real-time applications, edge deployment | ⭐⭐⭐⭐ |
| **Transformers** | Direct HuggingFace integration, flexibility | Research, custom models, prototyping | ⭐⭐⭐ |

## Quick Reference

See [INDEX.md](INDEX.md) for a complete matrix of all examples with their specifications.

## Choosing the Right Example

**If you want to...**

- **Learn the basics** → Start with `hf-llama-3_2-1b-instruct`
- **Optimize for throughput** → Try `vllm-gemma-3-1b-it` or `vllm-gemma-3-4b-it`
- **Minimize model size** → Use `sglang-smollm2-135m-instruct`
- **Add tool calling** → Check `vllm-tool-calling-llama-3.1-8b`
- **Build an AI agent** → Explore `agentic-gpt-oss-20b` or `agentic-gpt-5_1`
- **Generate embeddings** → See `vllm-embeddings` (if available)

## Common Configuration Options

### GPU Memory Requirements

- **8Gi**: Small models (135M - 1B params), suitable for development
- **20Gi**: Medium models (3B - 8B params), production-ready
- **48Gi**: Large models (20B+ params), advanced applications

### Model Classes

- **`ModelClass`**: Basic inference implementation
- **`OpenAIModelClass`**: OpenAI-compatible API endpoints (chat completions)
- **`AgenticModelClass`**: Advanced agentic workflows with tool integration

## Deployment Workflow

1. **Choose an example** based on your model size and complexity needs
2. **Review the README** in the example directory for specific requirements
3. **Customize config.yaml** with your model details and compute requirements
4. **Test locally** (optional): `clarifai model test`
5. **Deploy**: `clarifai model upload`
6. **Invoke**: Use Clarifai's API or web interface

## Performance Tips

- Use **vLLM** for maximum throughput with continuous batching
- Use **SGLang** for structured generation and prompt-heavy workloads
- Choose appropriate GPU memory to avoid OOM errors
- Enable streaming for better user experience in chat applications
- Consider quantization (GPTQ, AWQ) for larger models

## Additional Resources

- [Framework Index](../FRAMEWORK_INDEX.md) - Browse all examples by framework
- [Getting Started](../00-getting-started/) - Beginner tutorials
- [Clarifai Documentation](https://docs.clarifai.com/) - Platform documentation

## Contributing

When adding new LLM examples:
1. Place in appropriate complexity directory
2. Include README with frontmatter metadata
3. Follow naming convention: `framework-model-name`
4. Add comprehensive config.yaml with compute requirements
