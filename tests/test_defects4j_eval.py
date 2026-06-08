from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.embedding.hash_provider import HashEmbeddingProvider
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.storage.sqlite_store import SQLiteStore
from evals.defects4j.run_fault_localization_eval import evaluate_localization_case


def test_fault_localization_eval_scores_expected_file_and_methods(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "NumberUtils.java").write_text(
        """
package org.example;

public class NumberUtils {
    public int createNumber(String value) {
        return createInteger(value);
    }

    public int createInteger(String value) {
        return Integer.decode(value);
    }
}
""".strip(),
        encoding="utf-8",
    )

    settings = Settings(sqlite_path=tmp_path / "codeatlas.db")
    embedding = HashEmbeddingProvider()
    RepositoryIndexer(settings=settings, enable_qdrant=False, embedding_generator=embedding).index(repo)

    case = {
        "bug_id": "Lang_1b",
        "query": "NumberFormatException createNumber hexadecimal parsing",
        "impact_symbol": "createInteger",
        "expected_files": ["NumberUtils.java"],
        "expected_methods": [
            "org.example.NumberUtils.createNumber",
            "org.example.NumberUtils.createInteger",
        ],
    }

    result = evaluate_localization_case(
        SQLiteStore(settings.sqlite_path),
        embedding,
        settings,
        repo,
        case,
        raw_context_tokens=1000,
        limit=10,
        include_vectors=False,
        window_radius=2,
    )

    assert result["metrics"]["file_recall_at_5"] == 1.0
    assert result["metrics"]["method_recall_at_10"] == 1.0
    assert result["metrics"]["mrr"] > 0.0
    assert result["metrics"]["context_compression_ratio"] > 1.0
