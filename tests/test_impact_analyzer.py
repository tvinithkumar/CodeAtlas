from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.embedding.hash_provider import HashEmbeddingProvider
from codeatlas.graph.impact_analyzer import ImpactAnalyzer
from codeatlas.indexing.repository_indexer import RepositoryIndexer


def test_impact_analyzer_returns_callers_files_risks_and_windows(tmp_path: Path) -> None:
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
    (repo / "NumberUtilsTest.java").write_text(
        """
package org.example;

public class NumberUtilsTest {
    public void testCreateNumber() {
        new NumberUtils().createNumber("80000000");
    }
}
""".strip(),
        encoding="utf-8",
    )

    settings = Settings(sqlite_path=tmp_path / "codeatlas.db")
    indexer = RepositoryIndexer(
        settings=settings,
        enable_qdrant=False,
        embedding_generator=HashEmbeddingProvider(),
    )
    result = indexer.index(repo)

    assert result["edges"] > 0

    impact = ImpactAnalyzer(indexer.sqlite_store).analyze(
        "createInteger",
        repo_path=repo,
        depth=2,
        limit=10,
        window_radius=2,
    )

    assert impact is not None
    assert impact["symbol"]["qualified_name"] == "org.example.NumberUtils.createInteger"
    assert any(
        edge["src_symbol"] == "org.example.NumberUtils.createNumber"
        for edge in impact["direct_callers"]
    )
    assert any(item["file_path"] == "NumberUtils.java" for item in impact["files_affected"])
    assert impact["risk_notes"]
    assert impact["top_code_windows"]
    assert impact["top_code_windows"][0]["reason"] == "target definition"
