from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.enrichment.embedding_generator import HashEmbeddingGenerator
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.retrieval.hybrid_search import HybridSearch


def test_indexes_java_symbols_and_searches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "RetryConfig.java").write_text(
        """
package org.example;

public class RetryConfig {
    private static final int RETRY_BACKOFF_MS = 250;

    public RetryConfig() {
    }

    public int retryDelay(int attempt) {
        return RETRY_BACKOFF_MS * attempt;
    }
}
""".strip(),
        encoding="utf-8",
    )

    settings = Settings(sqlite_path=tmp_path / "codeatlas.db")
    indexer = RepositoryIndexer(
        settings=settings,
        enable_qdrant=False,
        embedding_generator=HashEmbeddingGenerator(),
    )
    result = indexer.index(repo)

    assert result["files"] == 1
    assert result["symbols"] == 4
    assert result["chunks"] == 5

    rows = indexer.sqlite_store.connection.execute(
        "SELECT qualified_name, kind, language FROM symbols ORDER BY qualified_name"
    ).fetchall()
    symbols = {(row["qualified_name"], row["kind"], row["language"]) for row in rows}

    assert ("org.example.RetryConfig", "class", "java") in symbols
    assert ("org.example.RetryConfig.RETRY_BACKOFF_MS", "field", "java") in symbols
    assert ("org.example.RetryConfig.RetryConfig", "constructor", "java") in symbols
    assert ("org.example.RetryConfig.retryDelay", "method", "java") in symbols

    search = HybridSearch(indexer.sqlite_store, indexer.embedding_generator, repo_path=repo)
    hits = search.search("retryDelay", include_vectors=False)

    assert hits
    assert any(hit.symbol == "org.example.RetryConfig.retryDelay" for hit in hits)
