#!/usr/bin/env python3
"""
Reorganize runners-examples repository structure.
Implements the migration from flat structure to complexity-based organization.
"""

import os
import shutil
from pathlib import Path

# Mapping of current paths to new paths
MOVES = {
    # Getting started
    "hello-world": "00-getting-started/hello-world",

    # LLM examples by complexity
    # Beginner (simple models, basic inference)
    "llm/hf-llama-3_2-1b-instruct": "llm/01-beginner/hf-llama-3_2-1b-instruct",
    "llm/vllm-phi-3.5-mini-instruct": "llm/01-beginner/vllm-phi-3.5-mini-instruct",
    "llm/Sglang-SmolLM2-135M-Instruct": "llm/01-beginner/sglang-smollm2-135m-instruct",

    # Intermediate (optimized inference, multiple frameworks)
    "llm/vllm-gemma-3-1b-it": "llm/02-intermediate/vllm-gemma-3-1b-it",
    "llm/vllm-gemma-3-4b-it": "llm/02-intermediate/vllm-gemma-3-4b-it",
    "llm/lmdeploy-Llama-3_2-3B-Instruct": "llm/02-intermediate/lmdeploy-llama-3_2-3b-instruct",
    "llm/vllm-embeddings": "llm/02-intermediate/vllm-embeddings",

    # Advanced (tool calling, agentic workflows)
    "llm/vllm-tool-calling-llama-3.1-8b": "llm/03-advanced/vllm-tool-calling-llama-3.1-8b",
    "llm/agentic-gpt-oss-20b": "llm/03-advanced/agentic-gpt-oss-20b",
    "llm/agentic-gpt-5_1": "llm/03-advanced/agentic-gpt-5_1",

    # Multimodal models - organize by type
    "multimodal-models/qwen2_5-VL-3B-Instruct-vllm": "multimodal-models/vision-language/qwen2_5-vl-3b-instruct-vllm",
    "multimodal-models/deepseek-ocr-sglang": "multimodal-models/ocr/deepseek-ocr-sglang",
    "ocr/nanonets-ocr-s": "multimodal-models/ocr/nanonets-ocr-s",

    # MCP servers by complexity
    # Basic (simple tools, minimal setup)
    "mcp/browser-mcp-server": "mcp-servers/01-basic/browser-mcp-server",
    "mcp/math": "mcp-servers/01-basic/math",
    "mcp/web-search": "mcp-servers/01-basic/web-search",

    # Integration (external APIs, services)
    "mcp/github-mcp-server": "mcp-servers/02-integration/github-mcp-server",
    "mcp/postgres": "mcp-servers/02-integration/postgres",
    "mcp/google-drive": "mcp-servers/02-integration/google-drive",
    "mcp/slack-tools-server": "mcp-servers/02-integration/slack-tools-server",

    # Advanced (complex features, docker)
    "mcp/code-execution-docker-version": "mcp-servers/03-advanced/code-execution-docker-version",
    "mcp/code-execution-without-docker-version": "mcp-servers/03-advanced/code-execution-without-docker-version",
    "mcp/browser-tools": "mcp-servers/03-advanced/browser-tools",
    "mcp/firecrawl-browser-tools": "mcp-servers/03-advanced/firecrawl-browser-tools",
}

def create_backup(repo_path: Path):
    """Create a backup of the current structure."""
    backup_path = repo_path / "backup"
    if backup_path.exists():
        print(f"⚠️  Backup already exists at {backup_path}")
        return

    print("📦 Creating backup...")
    backup_path.mkdir(exist_ok=True)

    # Backup critical directories
    for dir_name in ["llm", "mcp", "multimodal-models", "ocr", "hello-world"]:
        src = repo_path / dir_name
        if src.exists():
            dst = backup_path / dir_name
            shutil.copytree(src, dst)
            print(f"   ✓ Backed up {dir_name}/")

    print("✅ Backup complete\n")

def create_directory_structure(repo_path: Path):
    """Create the new directory structure."""
    print("📁 Creating new directory structure...")

    directories = [
        "00-getting-started",
        "llm/01-beginner",
        "llm/02-intermediate",
        "llm/03-advanced",
        "multimodal-models/vision-language",
        "multimodal-models/ocr",
        "mcp-servers/01-basic",
        "mcp-servers/02-integration",
        "mcp-servers/03-advanced",
        "scripts",
    ]

    for dir_path in directories:
        full_path = repo_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {dir_path}/")

    print("✅ Directory structure created\n")

def move_examples(repo_path: Path):
    """Move examples to their new locations."""
    print("📦 Moving examples...")

    for old_path, new_path in MOVES.items():
        src = repo_path / old_path
        dst = repo_path / new_path

        if not src.exists():
            print(f"   ⚠️  Skipping {old_path} (not found)")
            continue

        if dst.exists():
            print(f"   ⚠️  Skipping {old_path} (destination exists)")
            continue

        # Ensure parent directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Move the directory
        shutil.move(str(src), str(dst))
        print(f"   ✓ Moved {old_path} → {new_path}")

    print("✅ Examples moved\n")

def cleanup_empty_directories(repo_path: Path):
    """Remove empty directories after migration."""
    print("🧹 Cleaning up empty directories...")

    # Check and remove if empty
    empty_dirs = ["ocr", "mcp"]

    for dir_name in empty_dirs:
        dir_path = repo_path / dir_name
        if dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            print(f"   ✓ Removed empty {dir_name}/")

    print("✅ Cleanup complete\n")

def main():
    """Main migration function."""
    repo_path = Path("/Users/arman/work/runners-examples")

    print("\n" + "="*60)
    print("  Runners-Examples Repository Reorganization")
    print("="*60 + "\n")

    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        return 1

    # Phase 1: Backup
    create_backup(repo_path)

    # Phase 2: Create new structure
    create_directory_structure(repo_path)

    # Phase 3: Move examples
    move_examples(repo_path)

    # Phase 4: Cleanup
    cleanup_empty_directories(repo_path)

    print("="*60)
    print("✅ Migration complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python scripts/add_frontmatter.py")
    print("2. Run: python scripts/generate_indices.py")
    print("3. Review changes with: git status")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
