#!/usr/bin/env python3
"""
Add YAML frontmatter to all example README files.
Creates README files where missing and adds metadata for indexing.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Example metadata database
EXAMPLE_METADATA = {
    # Getting Started
    "00-getting-started/hello-world": {
        "complexity": "beginner",
        "framework": "custom",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["basic-inference"],
        "model_class": "ModelClass",
        "task": "hello-world",
    },

    # LLM - Beginner
    "llm/01-beginner/hf-llama-3_2-1b-instruct": {
        "complexity": "beginner",
        "framework": "transformers",
        "model_size": "1B",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["text-generation", "chat"],
        "model_class": "ModelClass",
        "task": "text-generation",
    },
    "llm/01-beginner/vllm-phi-3.5-mini-instruct": {
        "complexity": "beginner",
        "framework": "vllm",
        "model_size": "3.8B",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["text-generation", "chat", "streaming"],
        "model_class": "OpenAIModelClass",
        "task": "text-generation",
    },
    "llm/01-beginner/sglang-smollm2-135m-instruct": {
        "complexity": "beginner",
        "framework": "sglang",
        "model_size": "135M",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["text-generation", "chat", "lightweight"],
        "model_class": "OpenAIModelClass",
        "task": "text-generation",
    },

    # LLM - Intermediate
    "llm/02-intermediate/vllm-gemma-3-1b-it": {
        "complexity": "intermediate",
        "framework": "vllm",
        "model_size": "1B",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["text-generation", "chat", "streaming"],
        "model_class": "OpenAIModelClass",
        "task": "text-generation",
    },
    "llm/02-intermediate/vllm-gemma-3-4b-it": {
        "complexity": "intermediate",
        "framework": "vllm",
        "model_size": "4B",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["text-generation", "chat", "streaming"],
        "model_class": "OpenAIModelClass",
        "task": "text-generation",
    },
    "llm/02-intermediate/lmdeploy-llama-3_2-3b-instruct": {
        "complexity": "intermediate",
        "framework": "lmdeploy",
        "model_size": "3B",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["text-generation", "chat", "streaming"],
        "model_class": "OpenAIModelClass",
        "task": "text-generation",
    },
    "llm/02-intermediate/vllm-embeddings": {
        "complexity": "intermediate",
        "framework": "vllm",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["embeddings"],
        "model_class": "OpenAIModelClass",
        "task": "text-embedding",
    },

    # LLM - Advanced
    "llm/03-advanced/vllm-tool-calling-llama-3.1-8b": {
        "complexity": "advanced",
        "framework": "vllm",
        "model_size": "8B",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["text-generation", "tool-calling", "streaming"],
        "model_class": "OpenAIModelClass",
        "task": "text-generation",
    },
    "llm/03-advanced/agentic-gpt-oss-20b": {
        "complexity": "advanced",
        "framework": "vllm",
        "model_size": "20B",
        "gpu_required": True,
        "min_gpu_memory": "48Gi",
        "features": ["agentic", "tool-calling", "mcp-integration"],
        "model_class": "AgenticModelClass",
        "task": "text-generation",
    },
    "llm/03-advanced/agentic-gpt-5_1": {
        "complexity": "advanced",
        "framework": "vllm",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "48Gi",
        "features": ["agentic", "tool-calling", "mcp-integration"],
        "model_class": "AgenticModelClass",
        "task": "text-generation",
    },

    # Multimodal - Vision-Language
    "multimodal-models/vision-language/qwen2_5-vl-3b-instruct-vllm": {
        "complexity": "intermediate",
        "framework": "vllm",
        "model_size": "3B",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["vision-language", "chat", "streaming"],
        "model_class": "OpenAIModelClass",
        "task": "vision-language",
    },

    # Multimodal - OCR
    "multimodal-models/ocr/deepseek-ocr-sglang": {
        "complexity": "intermediate",
        "framework": "sglang",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["ocr", "vision-language"],
        "model_class": "OpenAIModelClass",
        "task": "ocr",
    },
    "multimodal-models/ocr/nanonets-ocr-s": {
        "complexity": "intermediate",
        "framework": "custom",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["ocr"],
        "model_class": "ModelClass",
        "task": "ocr",
    },

    # MCP - Basic
    "mcp-servers/01-basic/browser-mcp-server": {
        "complexity": "beginner",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "browser", "search"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/01-basic/math": {
        "complexity": "beginner",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "calculator"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/01-basic/web-search": {
        "complexity": "beginner",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "search"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },

    # MCP - Integration
    "mcp-servers/02-integration/github-mcp-server": {
        "complexity": "intermediate",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "github", "api-integration"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/02-integration/postgres": {
        "complexity": "intermediate",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "database", "postgresql"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/02-integration/google-drive": {
        "complexity": "intermediate",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "google-drive", "api-integration"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/02-integration/slack-tools-server": {
        "complexity": "intermediate",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "slack", "api-integration"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },

    # MCP - Advanced
    "mcp-servers/03-advanced/code-execution-docker-version": {
        "complexity": "advanced",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "code-execution", "docker", "sandbox"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/03-advanced/code-execution-without-docker-version": {
        "complexity": "advanced",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "code-execution"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/03-advanced/browser-tools": {
        "complexity": "advanced",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "browser", "automation"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },
    "mcp-servers/03-advanced/firecrawl-browser-tools": {
        "complexity": "advanced",
        "framework": "fastmcp",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["mcp", "browser", "web-scraping"],
        "model_class": "StdioMCPModelClass",
        "task": "mcp-server",
    },

    # Other categories
    "image-classifier/nsfw-image-classifier": {
        "complexity": "intermediate",
        "framework": "custom",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["image-classification", "nsfw-detection"],
        "model_class": "ModelClass",
        "task": "image-classification",
    },
    "image-detector/detr-resnet-image-detection": {
        "complexity": "intermediate",
        "framework": "transformers",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["object-detection"],
        "model_class": "ModelClass",
        "task": "object-detection",
    },
    "image-detector/dfine": {
        "complexity": "intermediate",
        "framework": "custom",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["object-detection"],
        "model_class": "ModelClass",
        "task": "object-detection",
    },
    "image-segmenter/mask2former-ade": {
        "complexity": "intermediate",
        "framework": "transformers",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["image-segmentation"],
        "model_class": "ModelClass",
        "task": "image-segmentation",
    },
    "text-embedder/jina-embeddings-v3": {
        "complexity": "intermediate",
        "framework": "transformers",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "8Gi",
        "features": ["embeddings"],
        "model_class": "ModelClass",
        "task": "text-embedding",
    },
    "text-to-image/flux-schnell": {
        "complexity": "intermediate",
        "framework": "custom",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["text-to-image", "diffusion"],
        "model_class": "ModelClass",
        "task": "text-to-image",
    },
    "image-text-to-image/stable-diffusion-2-depth": {
        "complexity": "intermediate",
        "framework": "custom",
        "model_size": "N/A",
        "gpu_required": True,
        "min_gpu_memory": "20Gi",
        "features": ["image-to-image", "depth-aware", "diffusion"],
        "model_class": "ModelClass",
        "task": "image-to-image",
    },
    "local-runners/ollama-model-upload": {
        "complexity": "beginner",
        "framework": "ollama",
        "model_size": "N/A",
        "gpu_required": False,
        "min_gpu_memory": "N/A",
        "features": ["local-development"],
        "model_class": "N/A",
        "task": "local-runner",
    },
}

README_TEMPLATE = """---
complexity: {complexity}
framework: {framework}
model_size: {model_size}
gpu_required: {gpu_required}
min_gpu_memory: {min_gpu_memory}
features: {features}
model_class: {model_class}
task: {task}
---

# {title}

{description}

## Quick Start

```bash
# Deploy this model
clarifai model upload
```

## Configuration

See `config.yaml` for model configuration details.

## Requirements

- GPU: {gpu_required}
- Minimum GPU Memory: {min_gpu_memory}
- Framework: {framework}

## Features

{features_list}
"""

def format_frontmatter(metadata: Dict) -> str:
    """Format metadata as YAML frontmatter."""
    features_str = "[" + ", ".join(metadata["features"]) + "]"

    return f"""---
complexity: {metadata['complexity']}
framework: {metadata['framework']}
model_size: {metadata['model_size']}
gpu_required: {str(metadata['gpu_required']).lower()}
min_gpu_memory: {metadata['min_gpu_memory']}
features: {features_str}
model_class: {metadata['model_class']}
task: {metadata['task']}
---

"""

def add_frontmatter_to_readme(readme_path: Path, metadata: Dict):
    """Add or update frontmatter in existing README."""
    if not readme_path.exists():
        return False

    content = readme_path.read_text()

    # Check if frontmatter already exists
    if content.startswith("---"):
        # Update existing frontmatter
        parts = content.split("---", 2)
        if len(parts) >= 3:
            # Replace old frontmatter with new
            frontmatter = format_frontmatter(metadata)
            new_content = frontmatter + parts[2].lstrip("\n")
            readme_path.write_text(new_content)
            return True

    # Add new frontmatter
    frontmatter = format_frontmatter(metadata)
    new_content = frontmatter + "\n" + content
    readme_path.write_text(new_content)
    return True

def create_readme_with_frontmatter(readme_path: Path, example_name: str, metadata: Dict):
    """Create a new README with frontmatter."""
    title = example_name.split("/")[-1].replace("-", " ").title()

    # Format features list
    features_list = "\n".join([f"- {feature}" for feature in metadata["features"]])

    content = README_TEMPLATE.format(
        complexity=metadata['complexity'],
        framework=metadata['framework'],
        model_size=metadata['model_size'],
        gpu_required=str(metadata['gpu_required']).lower(),
        min_gpu_memory=metadata['min_gpu_memory'],
        features="[" + ", ".join(metadata["features"]) + "]",
        model_class=metadata['model_class'],
        task=metadata['task'],
        title=title,
        description=f"Example demonstrating {metadata['task']} using {metadata['framework']}.",
        features_list=features_list,
    )

    readme_path.write_text(content)

def main():
    """Add frontmatter to all example READMEs."""
    repo_path = Path("/Users/arman/work/runners-examples")

    print("\n" + "="*60)
    print("  Adding Frontmatter to Example READMEs")
    print("="*60 + "\n")

    updated = 0
    created = 0
    skipped = 0

    for example_path, metadata in EXAMPLE_METADATA.items():
        full_path = repo_path / example_path
        readme_path = full_path / "README.md"

        if not full_path.exists():
            print(f"⚠️  Skipping {example_path} (directory not found)")
            skipped += 1
            continue

        if readme_path.exists():
            if add_frontmatter_to_readme(readme_path, metadata):
                print(f"✓ Updated {example_path}/README.md")
                updated += 1
            else:
                print(f"⚠️  Failed to update {example_path}/README.md")
                skipped += 1
        else:
            create_readme_with_frontmatter(readme_path, example_path, metadata)
            print(f"✓ Created {example_path}/README.md")
            created += 1

    print("\n" + "="*60)
    print(f"✅ Frontmatter addition complete!")
    print(f"   Updated: {updated}")
    print(f"   Created: {created}")
    print(f"   Skipped: {skipped}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
