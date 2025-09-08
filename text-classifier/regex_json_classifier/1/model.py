from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Concept
import sys
import os
sys.path.append(os.path.dirname(__file__))
from json_utils import json_candidates, safe_json_load, deep_collect
from regex_processing_utils import regex_scan, dedupe_sort


class TextClassifierModel(ModelClass):
    def load_model(self):
        pass

    @ModelClass.method
    def predict(self, prompt: str) -> list[Concept]:
        """Parse prompt for nested concept objects.

        Returns list[{'id','name','value'}]; returns [] if none parsed.
        """
        collected = []
        root = safe_json_load(prompt)
        if root is not None:
            deep_collect(root, collected)
        if not collected:
            for cand in json_candidates(prompt):
                parsed = safe_json_load(cand)
                if parsed is None:
                    continue
                deep_collect(parsed, collected)
                if collected:
                    break
        # Regex fallback
        if not collected:
            collected = regex_scan(prompt)
        if collected:
            return dedupe_sort(collected)
        # No concepts parsed
        return []
