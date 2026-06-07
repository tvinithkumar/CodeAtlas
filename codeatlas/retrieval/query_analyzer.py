from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    ripgrep_weight: float
    fts_weight: float
    vector_weight: float


class QueryAnalyzer:
    def classify(self, query: str) -> QueryPlan:
        if self._looks_exact(query):
            return QueryPlan(ripgrep_weight=0.70, fts_weight=0.20, vector_weight=0.10)
        return QueryPlan(ripgrep_weight=0.25, fts_weight=0.35, vector_weight=0.40)

    def _looks_exact(self, query: str) -> bool:
        patterns = [
            r"\b[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b",
            r"\b[a-z]+(?:_[a-z0-9]+)+\b",
            r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b",
            r"/[A-Za-z0-9_./{}:-]+",
            r"\b[A-Z_][A-Z0-9_]{2,}\b",
            r"\b[A-Za-z]+Error\b",
            r"\b[A-Za-z]+Exception\b",
            r"\b[A-Za-z0-9_]+\.[A-Za-z0-9_.]+\b",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

