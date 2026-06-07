from __future__ import annotations

from pathlib import Path
from typing import Any

from codeatlas.common.config import Settings
from codeatlas.enrichment.embedding_generator import HashEmbeddingGenerator
from codeatlas.mcp.limits import MAX_LINES_PER_RESULT, MAX_RESULTS, MAX_TOTAL_CHARS
from codeatlas.retrieval.code_window import CodeWindowFetcher
from codeatlas.retrieval.graph_search import GraphSearch
from codeatlas.retrieval.hybrid_search import HybridSearch
from codeatlas.storage.sqlite_store import SQLiteStore


class CodeAtlasMCPTools:
    def __init__(
        self,
        repo_path: str | Path,
        sqlite_path: str | Path = Settings().sqlite_path,
        max_results: int = MAX_RESULTS,
        max_lines_per_result: int = MAX_LINES_PER_RESULT,
        max_total_chars: int = MAX_TOTAL_CHARS,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.sqlite_store = SQLiteStore(Path(sqlite_path))
        self.max_results = max_results
        self.max_lines_per_result = max_lines_per_result
        self.max_total_chars = max_total_chars
        self.graph_search = GraphSearch(self.sqlite_store)

    def search_code(self, query: str, top_k: int = MAX_RESULTS) -> dict[str, Any]:
        limit = min(top_k, self.max_results)
        search = HybridSearch(
            self.sqlite_store,
            HashEmbeddingGenerator(),
            repo_path=self.repo_path,
        )
        hits = search.search(query, limit=limit, include_vectors=False)
        results = [
            self._limit_hit(
                {
                    "file_path": hit.file_path,
                    "symbol": hit.symbol,
                    "line_start": hit.line_start,
                    "line_end": hit.line_end,
                    "retrieval_method": hit.retrieval_method,
                    "content": hit.content,
                }
            )
            for hit in hits[:limit]
        ]
        return self._limit_response({"query": query, "results": results})

    def get_code_window(self, file: str, line: int, radius: int = 20) -> dict[str, Any]:
        radius = min(radius, self.max_lines_per_result // 2)
        window = CodeWindowFetcher().get_code_window(self.repo_path, file, line, radius)
        return self._limit_response(
            {
                "file_path": window.file_path,
                "line_start": window.line_start,
                "line_end": window.line_end,
                "content": self._limit_lines(window.content),
            }
        )

    def explain_symbol(self, symbol: str) -> dict[str, Any]:
        return self._limit_response(self.graph_search.explain_symbol(symbol) or {"symbol": symbol, "edges": []})

    def find_usages(self, symbol: str) -> dict[str, Any]:
        usages = self.graph_search.find_usage(symbol)[: self.max_results]
        return self._limit_response({"symbol": symbol, "usages": usages})

    def related_symbols(self, symbol: str) -> dict[str, Any]:
        related = self.graph_search.related_symbols(symbol, limit=self.max_results)
        return self._limit_response(related or {"symbol": symbol, "related_symbol_ids": [], "edges": []})

    def _limit_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        content = str(hit.get("content", ""))
        hit["content"] = self._limit_lines(content)
        return hit

    def _limit_lines(self, content: str) -> str:
        lines = content.splitlines()
        if len(lines) <= self.max_lines_per_result:
            return content
        return "\n".join(lines[: self.max_lines_per_result])

    def _limit_response(self, response: dict[str, Any]) -> dict[str, Any]:
        text = str(response)
        if len(text) <= self.max_total_chars:
            return response
        return {"truncated": True, "preview": text[: self.max_total_chars]}

