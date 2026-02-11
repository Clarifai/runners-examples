#!/usr/bin/env python3
"""
Flatten the structure by removing complexity subdirectories.
Move examples directly under their category directories.
"""

import shutil
from pathlib import Path

repo_path = Path("/Users/arman/work/runners-examples")

# LLM examples - move from complexity subdirs to llm/
llm_examples = list((repo_path / "llm").rglob("*/config.yaml"))
for config_file in llm_examples:
    example_dir = config_file.parent
    if "/01-" in str(example_dir) or "/02-" in str(example_dir) or "/03-" in str(example_dir):
        target = repo_path / "llm" / example_dir.name
        if not target.exists():
            shutil.move(str(example_dir), str(target))
            print(f"✓ Moved {example_dir.relative_to(repo_path)} → {target.relative_to(repo_path)}")

# MCP servers - move from complexity subdirs to mcp-servers/
mcp_examples = list((repo_path / "mcp-servers").rglob("*/config.yaml"))
for config_file in mcp_examples:
    example_dir = config_file.parent
    if "/01-" in str(example_dir) or "/02-" in str(example_dir) or "/03-" in str(example_dir):
        target = repo_path / "mcp-servers" / example_dir.name
        if not target.exists():
            shutil.move(str(example_dir), str(target))
            print(f"✓ Moved {example_dir.relative_to(repo_path)} → {target.relative_to(repo_path)}")

# Remove empty complexity subdirectories
for complexity_dir in ["01-beginner", "02-intermediate", "03-advanced", "01-basic", "02-integration"]:
    for parent in ["llm", "mcp-servers"]:
        dir_path = repo_path / parent / complexity_dir
        if dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            print(f"✓ Removed empty {dir_path.relative_to(repo_path)}")

# Also handle 00-getting-started - move hello-world up
getting_started = repo_path / "00-getting-started" / "hello-world"
if getting_started.exists():
    target = repo_path / "hello-world"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(getting_started), str(target))
    print(f"✓ Moved 00-getting-started/hello-world → hello-world")
    
    # Remove 00-getting-started if empty
    gs_dir = repo_path / "00-getting-started"
    if gs_dir.exists():
        # Remove README too
        readme = gs_dir / "README.md"
        if readme.exists():
            readme.unlink()
        if not any(gs_dir.iterdir()):
            gs_dir.rmdir()
            print(f"✓ Removed 00-getting-started/")

print("\n✅ Structure flattened!")
