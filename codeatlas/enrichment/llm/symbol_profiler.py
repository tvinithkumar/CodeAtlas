from __future__ import annotations

from codeatlas.chunking.models import CodeChunk
from codeatlas.enrichment.llm.enricher import LLMEnricher
from codeatlas.enrichment.llm.models import LLMEnrichment
from codeatlas.symbols.models import Symbol


class LLMSymbolProfiler:
    def __init__(self, enricher: LLMEnricher) -> None:
        self.enricher = enricher

    def profile(self, symbol: Symbol, chunk: CodeChunk) -> LLMEnrichment:
        return self.enricher.enrich(symbol, chunk)
