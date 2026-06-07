from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.enrichment.embedding_generator import HashEmbeddingGenerator
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.mcp.tools import CodeAtlasMCPTools


def _indexed_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        """
RETRY_BACKOFF_MS = 250

def retry_delay(attempt: int) -> int:
    return RETRY_BACKOFF_MS * attempt

def handler() -> int:
    return retry_delay(2)
""".strip(),
        encoding="utf-8",
    )
    sqlite_path = tmp_path / "codeatlas.db"
    indexer = RepositoryIndexer(
        settings=Settings(sqlite_path=sqlite_path),
        enable_qdrant=False,
        embedding_generator=HashEmbeddingGenerator(),
    )
    indexer.index(repo)
    return repo, sqlite_path


def test_mcp_tools_search_window_and_graph(tmp_path: Path) -> None:
    repo, sqlite_path = _indexed_repo(tmp_path)
    tools = CodeAtlasMCPTools(repo, sqlite_path=sqlite_path, max_results=3, max_lines_per_result=5)

    search = tools.search_code("RETRY_BACKOFF_MS", top_k=10)
    assert len(search["results"]) <= 3
    assert any("RETRY_BACKOFF_MS" in result["content"] for result in search["results"])

    window = tools.get_code_window("service.py", line=4, radius=10)
    assert window["line_end"] - window["line_start"] + 1 <= 5

    explanation = tools.explain_symbol("retry_delay")
    assert explanation["symbol"]["qualified_name"] == "retry_delay"

    usages = tools.find_usages("retry_delay")
    assert usages["usages"]

    related = tools.related_symbols("RETRY_BACKOFF_MS")
    assert related["edges"]


def test_mcp_tools_apply_total_char_limit(tmp_path: Path) -> None:
    repo, sqlite_path = _indexed_repo(tmp_path)
    tools = CodeAtlasMCPTools(repo, sqlite_path=sqlite_path, max_total_chars=20)

    response = tools.search_code("RETRY_BACKOFF_MS")

    assert response["truncated"] is True
    assert len(response["preview"]) <= 20

