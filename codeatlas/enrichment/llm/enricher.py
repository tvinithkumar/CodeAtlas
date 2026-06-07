from __future__ import annotations

from codeatlas.chunking.models import CodeChunk
from codeatlas.enrichment.llm.base import LLMClient
from codeatlas.enrichment.llm.json_parser import LLMProfileJSONParser
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
        self.parser = LLMProfileJSONParser()

    def enrich(self, symbol: Symbol, chunk: CodeChunk) -> LLMEnrichment:
        prompts = [
            describe_symbol_prompt(symbol, chunk),
            summarize_function_prompt(symbol, chunk),
            generate_search_tags_prompt(symbol, chunk),
        ]
        enrichments = [self._complete(prompt) for prompt in prompts]
        return self._merge(enrichments)

    def _complete(self, prompt: str) -> LLMEnrichment:
        try:
            raw = self.client.complete(prompt)
        except Exception:
            return LLMEnrichment()
        return self.parser.parse(raw)

    def _merge(self, enrichments: list[LLMEnrichment]) -> LLMEnrichment:
        description = next((item.description for item in enrichments if item.description), "")
        return LLMEnrichment(
            description=description,
            responsibilities=self._unique(item for enrichment in enrichments for item in enrichment.responsibilities),
            inputs=self._unique(item for enrichment in enrichments for item in enrichment.inputs),
            outputs=self._unique(item for enrichment in enrichments for item in enrichment.outputs),
            side_effects=self._unique(item for enrichment in enrichments for item in enrichment.side_effects),
            failure_modes=self._unique(item for enrichment in enrichments for item in enrichment.failure_modes),
            search_tags=self._unique(item for enrichment in enrichments for item in enrichment.search_tags),
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
