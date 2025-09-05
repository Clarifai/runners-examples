from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Concept
from typing import List, Any
import re
import json


class TextClassifierModel(ModelClass):
    def load_model(self):
        pass

    # --- Parsing helpers ---
    @staticmethod
    def _json_candidates(text: str):
        # fenced blocks
        for m in re.finditer(r"```json\s*(.+?)```", text, re.IGNORECASE | re.DOTALL):
            yield m.group(1)
        # brute-force find arrays / objects that look like they contain name/value
        for m in re.finditer(r"(\{[^{}]*?\"name\"[^{}]*?\}|\[[^\]]+\])", text, re.DOTALL):
            frag = m.group(1)
            if '"name"' in frag and '"value"' in frag:
                yield frag

    @staticmethod
    def _safe_json_load(s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            return None

    @classmethod
    def _deep_collect(cls, node: Any, bucket: List[dict]):
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
                cls._deep_collect(v, bucket)
        elif isinstance(node, list):
            for item in node:
                cls._deep_collect(item, bucket)

    @staticmethod
    def _regex_scan(text: str):
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

    @staticmethod
    def _dedupe_sort(concepts: List[dict]):
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

    @ModelClass.method
    def predict(self, prompt: str) -> list[Concept]:
        """Parse prompt for nested concept objects.

        Returns list[{'id','name','value'}]; returns [] if none parsed.
        """
        collected = []
        root = self._safe_json_load(prompt)
        if root is not None:
            self._deep_collect(root, collected)
        if not collected:
            for cand in self._json_candidates(prompt):
                parsed = self._safe_json_load(cand)
                if parsed is None:
                    continue
                self._deep_collect(parsed, collected)
                if collected:
                    break
        # Regex fallback
        if not collected:
            collected = self._regex_scan(prompt)
        if collected:
            return self._dedupe_sort(collected)
        # No concepts parsed
        return []
