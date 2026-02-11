#!/usr/bin/env python3
"""
Generate index files by parsing README frontmatter.
Creates FRAMEWORK_INDEX.md and category-specific INDEX.md files.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def parse_frontmatter(readme_path: Path) -> Dict:
    """Extract YAML frontmatter from README."""
    if not readme_path.exists():
        return {}

    content = readme_path.read_text()

    if not content.startswith("---"):
        return {}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}

    try:
        frontmatter = yaml.safe_load(parts[1])
        return frontmatter or {}
    except yaml.YAMLError:
        return {}

def collect_all_examples(repo_path: Path) -> List[Tuple[str, Dict]]:
    """Collect all examples with their metadata."""
    examples = []

    # Find all README.md files in example directories
    for readme_path in repo_path.rglob("*/README.md"):
        # Skip root README and category READMEs
        if readme_path.parent == repo_path:
            continue
        if readme_path.name == "README.md" and not (readme_path.parent / "config.yaml").exists():
            continue

        # Parse frontmatter
        metadata = parse_frontmatter(readme_path)
        if not metadata:
            continue

        # Get relative path from repo root
        rel_path = readme_path.parent.relative_to(repo_path)

        examples.append((str(rel_path), metadata))

    return examples

def generate_framework_index(repo_path: Path, examples: List[Tuple[str, Dict]]):
    """Generate FRAMEWORK_INDEX.md at repository root."""
    print("📝 Generating FRAMEWORK_INDEX.md...")

    # Group by framework
    by_framework = defaultdict(list)
    for path, meta in examples:
        framework = meta.get("framework", "unknown")
        by_framework[framework].append((path, meta))

    # Sort examples within each framework by name
    for framework in by_framework:
        by_framework[framework].sort(key=lambda x: x[0])

    # Generate markdown
    lines = [
        "# Examples by Framework",
        "",
        "This index organizes all examples by the framework they use. "
        "Use this to find examples for your preferred deployment framework.",
        "",
    ]

    framework_display = {
        "vllm": "vLLM",
        "sglang": "SGLang",
        "lmdeploy": "LMDeploy",
        "transformers": "HuggingFace Transformers",
        "fastmcp": "FastMCP",
        "ollama": "Ollama",
        "custom": "Custom Implementation",
    }

    for framework in sorted(by_framework.keys()):
        display_name = framework_display.get(framework, framework.title())
        lines.append(f"## {display_name}")
        lines.append("")

        for path, meta in by_framework[framework]:
            name = path.split("/")[-1]
            model_size = meta.get("model_size", "N/A")
            gpu_mem = meta.get("min_gpu_memory", "N/A")
            features = meta.get("features", [])

            feature_str = ", ".join(features[:2])  # Show first 2 features
            if len(features) > 2:
                feature_str += f", +{len(features) - 2} more"

            lines.append(f"- [{name}]({path}/) - "
                        f"{model_size}, {gpu_mem} GPU, {feature_str}")

        lines.append("")

    content = "\n".join(lines)
    output_path = repo_path / "FRAMEWORK_INDEX.md"
    output_path.write_text(content)

    print(f"   ✓ Created FRAMEWORK_INDEX.md ({len(by_framework)} frameworks)")

def generate_llm_index(repo_path: Path, examples: List[Tuple[str, Dict]]):
    """Generate llm/INDEX.md with matrix view."""
    print("📝 Generating llm/INDEX.md...")

    # Filter LLM examples
    llm_examples = [(p, m) for p, m in examples if p.startswith("llm/")]

    if not llm_examples:
        return

    # Generate matrix table
    lines = [
        "# LLM Examples Index",
        "",
        "Quick reference for all LLM examples with their specifications.",
        "",
        "| Example | Framework | Size | GPU Memory | Key Features |",
        "|---------|-----------|------|------------|--------------|",
    ]

    for path, meta in sorted(llm_examples, key=lambda x: x[0]):
        name = path.split("/")[-1]
        framework = meta.get("framework", "N/A")
        model_size = meta.get("model_size", "N/A")
        gpu_mem = meta.get("min_gpu_memory", "N/A")
        features = meta.get("features", [])

        # Take top 2 features
        feature_str = ", ".join(features[:2])

        lines.append(f"| [{name}]({name}/) | {framework} | {model_size} | "
                    f"{gpu_mem} | {feature_str} |")

    lines.append("")
    lines.append("## By Framework")
    lines.append("")

    # Group by framework
    by_framework = defaultdict(list)
    for path, meta in llm_examples:
        framework = meta.get("framework", "unknown")
        by_framework[framework].append((path, meta))

    for framework in sorted(by_framework.keys()):
        if not by_framework[framework]:
            continue

        lines.append(f"### {framework.upper() if len(framework) <= 4 else framework.title()}")
        lines.append("")

        for path, meta in sorted(by_framework[framework], key=lambda x: x[0]):
            name = path.split("/")[-1]
            model_size = meta.get("model_size", "N/A")
            lines.append(f"- [{name}]({name}/) - {model_size}")

        lines.append("")

    content = "\n".join(lines)
    output_path = repo_path / "llm" / "INDEX.md"
    output_path.write_text(content)

    print(f"   ✓ Created llm/INDEX.md ({len(llm_examples)} examples)")

def generate_mcp_index(repo_path: Path, examples: List[Tuple[str, Dict]]):
    """Generate mcp-servers/INDEX.md with feature matrix."""
    print("📝 Generating mcp-servers/INDEX.md...")

    # Filter MCP examples
    mcp_examples = [(p, m) for p, m in examples if p.startswith("mcp-servers/")]

    if not mcp_examples:
        return

    # Detect common features
    all_features = set()
    for _, meta in mcp_examples:
        all_features.update(meta.get("features", []))

    # Pick key feature categories
    feature_columns = ["browser", "api-integration", "code-execution", "database"]

    lines = [
        "# MCP Servers Feature Matrix",
        "",
        "Overview of all MCP server examples with their capabilities.",
        "",
        "| Server | " + " | ".join([f.replace("-", " ").title() for f in feature_columns]) + " |",
        "|--------|" + "|".join(["--------"] * len(feature_columns)) + "|",
    ]

    for path, meta in sorted(mcp_examples, key=lambda x: x[0]):
        name = path.split("/")[-1]
        features = meta.get("features", [])

        # Check each feature column
        feature_marks = []
        for feature in feature_columns:
            has_feature = any(feature in f for f in features)
            feature_marks.append("✓" if has_feature else "-")

        lines.append(f"| [{name}]({name}/) | " +
                    " | ".join(feature_marks) + " |")

    lines.append("")
    lines.append("## All Servers")
    lines.append("")

    for path, meta in sorted(mcp_examples, key=lambda x: x[0]):
        name = path.split("/")[-1]
        features = ", ".join(meta.get("features", [])[:3])
        lines.append(f"- [{name}]({name}/) - {features}")

    lines.append("")

    content = "\n".join(lines)
    output_path = repo_path / "mcp-servers" / "INDEX.md"
    output_path.write_text(content)

    print(f"   ✓ Created mcp-servers/INDEX.md ({len(mcp_examples)} examples)")

def main():
    """Generate all index files."""
    repo_path = Path("/Users/arman/work/runners-examples")

    print("\n" + "="*60)
    print("  Generating Index Files")
    print("="*60 + "\n")

    # Collect all examples
    print("🔍 Scanning for examples...")
    examples = collect_all_examples(repo_path)
    print(f"   Found {len(examples)} examples with metadata\n")

    # Generate indices
    generate_framework_index(repo_path, examples)
    generate_llm_index(repo_path, examples)
    generate_mcp_index(repo_path, examples)

    print("\n" + "="*60)
    print("✅ Index generation complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
