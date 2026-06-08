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

import java.util.List;

class BaseConfig {
}

public class RetryConfig extends BaseConfig {
    private static final int RETRY_BACKOFF_MS = 250;

    public RetryConfig() {
    }

    public int retryDelay(int attempt) {
        return createNumber(RETRY_BACKOFF_MS) * attempt;
    }

    public int createNumber(int value) {
        return value;
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
    assert result["symbols"] == 6
    assert result["chunks"] == 7
    assert result["edges"] >= 4

    rows = indexer.sqlite_store.connection.execute(
        "SELECT qualified_name, kind, language FROM symbols ORDER BY qualified_name"
    ).fetchall()
    symbols = {(row["qualified_name"], row["kind"], row["language"]) for row in rows}

    assert ("org.example.BaseConfig", "class", "java") in symbols
    assert ("org.example.RetryConfig", "class", "java") in symbols
    assert ("org.example.RetryConfig.RETRY_BACKOFF_MS", "field", "java") in symbols
    assert ("org.example.RetryConfig.RetryConfig", "constructor", "java") in symbols
    assert ("org.example.RetryConfig.retryDelay", "method", "java") in symbols
    assert ("org.example.RetryConfig.createNumber", "method", "java") in symbols

    edges = indexer.sqlite_store.connection.execute(
        "SELECT src_symbol_id, dst_symbol_id, edge_type, line_start, line_end FROM symbol_edges"
    ).fetchall()
    edge_set = {(row["src_symbol_id"], row["dst_symbol_id"], row["edge_type"]) for row in edges}

    retry_class_id = next(row["src_symbol_id"] for row in edges if row["edge_type"] == "imports")
    base_config_id = next(row["dst_symbol_id"] for row in edges if row["edge_type"] == "inherits")
    create_number_id = indexer.sqlite_store.find_symbols("createNumber", limit=1)[0]["id"]
    retry_backoff_id = indexer.sqlite_store.find_symbols("RETRY_BACKOFF_MS", limit=1)[0]["id"]

    assert (retry_class_id, "java.util.List", "imports") in edge_set
    assert (indexer.sqlite_store.find_symbols("RetryConfig", limit=1)[0]["id"], base_config_id, "inherits") in edge_set
    assert any(row["dst_symbol_id"] == create_number_id and row["edge_type"] == "calls" for row in edges)
    assert any(row["dst_symbol_id"] == retry_backoff_id and row["edge_type"] == "uses" for row in edges)
    assert all(row["line_start"] is not None and row["line_end"] is not None for row in edges)

    search = HybridSearch(indexer.sqlite_store, indexer.embedding_generator, repo_path=repo)
    hits = search.search("retryDelay", include_vectors=False)

    assert hits
    assert any(hit.symbol == "org.example.RetryConfig.retryDelay" for hit in hits)
