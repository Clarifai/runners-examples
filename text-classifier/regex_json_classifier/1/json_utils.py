import re
import json
from typing import Any, List
from clarifai.runners.utils.data_types import Concept


def json_candidates(text: str):
    # fenced blocks
    for m in re.finditer(r"```json\s*(.+?)```", text, re.IGNORECASE | re.DOTALL):
        yield m.group(1)
    # brute-force find arrays / objects that look like they contain name/value
    for m in re.finditer(r"(\{[^{}]*?\"name\"[^{}]*?\}|\[[^\]]+\])", text, re.DOTALL):
        frag = m.group(1)
        if '"name"' in frag and '"value"' in frag:
            yield frag


def safe_json_load(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def deep_collect(node: Any, bucket: List[dict]):
    if isinstance(node, dict):
        # possible concept dict
        keys = node.keys()
        if 'name' in keys and 'value' in keys:
            name = node.get('name')
            value = node.get('value')
            id_val = node.get('id') or (Concept._concept_name_to_id(name) if name else None)
            try:
                if name is not None and value is not None:
                    value_f = float(value)
                    bucket.append({'id': id_val, 'name': name, 'value': value_f})
            except (TypeError, ValueError):
                pass
        # recurse
        for v in node.values():
            deep_collect(v, bucket)
    elif isinstance(node, list):
        for item in node:
            deep_collect(item, bucket)