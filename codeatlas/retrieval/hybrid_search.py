from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codeatlas.enrichment.embedding_generator import EmbeddingGenerator
from codeatlas.retrieval.fts_search import FTSSearch
from codeatlas.retrieval.query_analyzer import QueryAnalyzer
from codeatlas.retrieval.reranker import ReciprocalRankFusion
from codeatlas.retrieval.ripgrep_search import RipgrepRetriever
from codeatlas.retrieval.vector_search import VectorSearch
from codeatlas.storage.models import SearchHit
from codeatlas.storage.sqlite_store import SQLiteStore
from codeatlas.storage.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class HybridSearchResult:
    hits: list[SearchHit]
    raw_retrieval_method_counts: dict[str, int]
    errors: dict[str, str] = field(default_factory=dict)


class HybridSearch:
    def __init__(
        self,
        sqlite_store: SQLiteStore,
        embedding_generator: EmbeddingGenerator,
        vector_store: QdrantVectorStore | None = None,
        repo_path: str | Path | None = None,
        ripgrep_retriever: RipgrepRetriever | None = None,
    ) -> None:
        self.sqlite_store = sqlite_store
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.repo_path = Path(repo_path) if repo_path is not None else None
        self.fts_search = FTSSearch(sqlite_store)
        self.ripgrep_retriever = ripgrep_retriever or RipgrepRetriever()
        self.query_analyzer = QueryAnalyzer()
        self.reranker = ReciprocalRankFusion()

    def search(self, query: str, limit: int = 10, include_vectors: bool = True) -> list[SearchHit]:
        return self.search_detailed(query, limit=limit, include_vectors=include_vectors).hits

    def search_detailed(self, query: str, limit: int = 10, include_vectors: bool = True) -> HybridSearchResult:
        plan = self.query_analyzer.classify(query)
        ranked_lists: list[tuple[float, list[SearchHit]]] = []
        raw_counts: dict[str, int] = {}
        errors: dict[str, str] = {}

        if self.repo_path is not None:
            ripgrep_hits = self.ripgrep_retriever.search_hits(self.repo_path, query, max_results=limit)
            ranked_lists.append((plan.ripgrep_weight, ripgrep_hits))
            raw_counts["ripgrep"] = len(ripgrep_hits)

        fts_hits = self.fts_search.search(query, limit=limit)
        ranked_lists.append((plan.fts_weight, fts_hits))
        raw_counts["fts"] = len(fts_hits)
        if include_vectors and self.vector_store is not None:
            try:
                vector_search = VectorSearch(self.embedding_generator, self.vector_store)
                vector_hits = vector_search.search(query, limit=limit)
                ranked_lists.append((plan.vector_weight, vector_hits))
                raw_counts["vector"] = len(vector_hits)
            except Exception as exc:
                raw_counts["vector"] = 0
                errors["vector"] = str(exc)

        return HybridSearchResult(
            hits=self.reranker.fuse(ranked_lists, limit=limit),
            raw_retrieval_method_counts=raw_counts,
            errors=errors,
        )
