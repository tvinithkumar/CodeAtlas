from __future__ import annotations

import json
import re

from codeatlas.chunking.models import CodeChunk
from codeatlas.enrichment.llm.base import LLMClient
from codeatlas.enrichment.llm.models import LLMEnrichment
from codeatlas.enrichment.llm.prompts import (
    describe_symbol_prompt,
    generate_search_tags_prompt,
    summarize_function_prompt,
)
from codeatlas.symbols.models import Symbol


class LLMEnricher:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def enrich(self, symbol: Symbol, chunk: CodeChunk) -> LLMEnrichment:
        prompts = [
            describe_symbol_prompt(symbol, chunk),
            summarize_function_prompt(symbol, chunk),
            generate_search_tags_prompt(symbol, chunk),
        ]
        enrichments = [self._complete(prompt) for prompt in prompts]
        return self._merge(enrichments)

    def _complete(self, prompt: str) -> LLMEnrichment:
        raw = self.client.complete(prompt)
        data = self._parse_json(raw)
        return LLMEnrichment(
            description=str(data.get("description", "")).strip(),
            tags=self._string_list(data.get("tags", [])),
            inputs=self._string_list(data.get("inputs", [])),
            outputs=self._string_list(data.get("outputs", [])),
            failure_modes=self._string_list(data.get("failure_modes", [])),
        )

    def _parse_json(self, raw: str) -> dict[str, object]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _merge(self, enrichments: list[LLMEnrichment]) -> LLMEnrichment:
        description = next((item.description for item in enrichments if item.description), "")
        return LLMEnrichment(
            description=description,
            tags=self._unique(item for enrichment in enrichments for item in enrichment.tags),
            inputs=self._unique(item for enrichment in enrichments for item in enrichment.inputs),
            outputs=self._unique(item for enrichment in enrichments for item in enrichment.outputs),
            failure_modes=self._unique(item for enrichment in enrichments for item in enrichment.failure_modes),
        )

    def _unique(self, values) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

