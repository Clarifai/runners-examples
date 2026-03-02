#!/usr/bin/env python3
"""
Validate that all examples have the required structure and metadata.
Used by CI to ensure consistency.
"""

import sys
import yaml
from pathlib import Path
from typing import List, Tuple

REQUIRED_FILES = ["config.yaml", "README.md", "requirements.txt"]
REQUIRED_FRONTMATTER_FIELDS = [
    "complexity",
    "framework",
    "model_size",
    "gpu_required",
    "min_gpu_memory",
    "features",
    "model_class",
    "task",
]

def find_examples(repo_path: Path) -> List[Path]:
    """Find all example directories (those containing config.yaml)."""
    examples = []
    for config_file in repo_path.rglob("config.yaml"):
        example_dir = config_file.parent
        # Skip backup directory
        if "backup" in str(example_dir):
            continue
        # Skip script directories
        if "scripts" in str(example_dir):
            continue
        examples.append(example_dir)
    return sorted(examples)

def validate_required_files(example_dir: Path) -> Tuple[bool, List[str]]:
    """Check that all required files exist."""
    errors = []
    for file_name in REQUIRED_FILES:
        file_path = example_dir / file_name
        if not file_path.exists():
            errors.append(f"Missing required file: {file_name}")

    # Check for model.py in 1/ subdirectory
    model_py = example_dir / "1" / "model.py"
    if not model_py.exists():
        errors.append("Missing required file: 1/model.py")

    return len(errors) == 0, errors

def parse_frontmatter(readme_path: Path) -> Tuple[bool, dict, List[str]]:
    """Parse and validate README frontmatter."""
    errors = []

    if not readme_path.exists():
        return False, {}, ["README.md does not exist"]

    content = readme_path.read_text()

    if not content.startswith("---"):
        errors.append("README.md missing YAML frontmatter")
        return False, {}, errors

    parts = content.split("---", 2)
    if len(parts) < 3:
        errors.append("Invalid YAML frontmatter format")
        return False, {}, errors

    try:
        frontmatter = yaml.safe_load(parts[1])
        if not frontmatter:
            errors.append("Empty frontmatter")
            return False, {}, errors
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in frontmatter: {e}")
        return False, {}, errors

    # Check required fields
    for field in REQUIRED_FRONTMATTER_FIELDS:
        if field not in frontmatter:
            errors.append(f"Missing frontmatter field: {field}")

    # Validate complexity values
    if "complexity" in frontmatter:
        valid_complexity = ["beginner", "intermediate", "advanced"]
        if frontmatter["complexity"] not in valid_complexity:
            errors.append(f"Invalid complexity value: {frontmatter['complexity']} (must be one of {valid_complexity})")

    # Validate features is a list
    if "features" in frontmatter:
        if not isinstance(frontmatter["features"], list):
            errors.append("features field must be a list")

    return len(errors) == 0, frontmatter, errors

def validate_naming_convention(example_path: Path, repo_path: Path) -> Tuple[bool, List[str]]:
    """Check that example follows naming conventions."""
    errors = []
    rel_path = example_path.relative_to(repo_path)
    example_name = example_path.name

    # Check for lowercase and hyphens
    if example_name != example_name.lower():
        errors.append(f"Example name should be lowercase: {example_name}")

    # Check for spaces
    if " " in example_name:
        errors.append(f"Example name should not contain spaces: {example_name}")

    return len(errors) == 0, errors

def main():
    """Main validation function."""
    repo_path = Path("/Users/arman/work/runners-examples")

    print("\n" + "="*60)
    print("  Validating Repository Structure")
    print("="*60 + "\n")

    examples = find_examples(repo_path)
    print(f"Found {len(examples)} examples to validate\n")

    all_valid = True
    validation_results = []

    for example_dir in examples:
        rel_path = example_dir.relative_to(repo_path)
        errors = []

        # Validate required files
        files_valid, file_errors = validate_required_files(example_dir)
        errors.extend(file_errors)

        # Validate README frontmatter
        readme_path = example_dir / "README.md"
        fm_valid, frontmatter, fm_errors = parse_frontmatter(readme_path)
        errors.extend(fm_errors)

        # Validate naming convention
        naming_valid, naming_errors = validate_naming_convention(example_dir, repo_path)
        errors.extend(naming_errors)

        # Record results
        is_valid = len(errors) == 0
        validation_results.append((rel_path, is_valid, errors))

        if is_valid:
            print(f"✓ {rel_path}")
        else:
            print(f"✗ {rel_path}")
            for error in errors:
                print(f"  - {error}")
            all_valid = False

    print("\n" + "="*60)

    if all_valid:
        valid_count = len([r for r in validation_results if r[1]])
        print(f"✅ All {valid_count} examples are valid!")
        print("="*60 + "\n")
        return 0
    else:
        valid_count = len([r for r in validation_results if r[1]])
        invalid_count = len([r for r in validation_results if not r[1]])
        print(f"❌ Validation failed: {valid_count} valid, {invalid_count} invalid")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
