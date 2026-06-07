from __future__ import annotations

import json
import re
from typing import Any

from codeatlas.enrichment.llm.profile_schema import SymbolProfile


class LLMProfileJSONParser:
    def parse(self, raw: str) -> SymbolProfile:
        data = self._parse_object(raw)
        if not data:
            return SymbolProfile()
        return SymbolProfile(
            description=str(data.get("description", "")).strip(),
            responsibilities=self._string_list(data.get("responsibilities", [])),
            inputs=self._string_list(data.get("inputs", [])),
            outputs=self._string_list(data.get("outputs", [])),
            side_effects=self._string_list(data.get("side_effects", [])),
            failure_modes=self._string_list(data.get("failure_modes", [])),
            search_tags=self._string_list(data.get("search_tags", data.get("tags", []))),
        )

    def _parse_object(self, raw: str) -> dict[str, Any]:
        for candidate in self._candidates(raw):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return {}

    def _candidates(self, raw: str) -> list[str]:
        stripped = raw.strip()
        candidates = [stripped]

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0).strip())
        return candidates

    def _string_list(self, value: object) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

