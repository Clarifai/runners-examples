import re
from typing import List
from clarifai.runners.utils.data_types import Concept


def regex_scan(text: str):
    pattern = re.compile(
        r"\{[^{}]*?\"name\"\s*:\s*\"([^\"]+)\"[^{}]*?\"value\"\s*:\s*([0-9.eE+-]+)[^{}]*?\}"
    )
    out = []
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        out.append({'id': Concept._concept_name_to_id(name), 'name': name, 'value': val})
    return out


def dedupe_sort(concepts: List[dict]):
    seen = {}
    for c in concepts:
        key = (c.get('id') or c.get('name', '')).lower()
        if key not in seen or c.get('value', 0) > seen[key].get('value', 0):
            seen[key] = c
    ordered = list(seen.values())
    ordered.sort(key=lambda x: x.get('value', 0), reverse=True)
    concepts = [
        Concept(id=c.get('id'), name=c.get('name'), value=c.get('value')) for c in ordered
    ]
    return concepts