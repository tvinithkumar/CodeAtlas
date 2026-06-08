from __future__ import annotations

from pathlib import Path
import shutil

from codeatlas.common.config import Settings
from codeatlas.enrichment.embedding_generator import HashEmbeddingGenerator
from codeatlas.enrichment.llm.enricher import LLMEnricher
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.retrieval.graph_search import GraphSearch
from codeatlas.retrieval.hybrid_search import HybridSearch
from codeatlas.retrieval.query_analyzer import QueryAnalyzer
from codeatlas.retrieval.ripgrep_search import RipgrepRetriever


class FakeLLMClient:
    def complete(self, prompt: str) -> str:
        return """
{
  "description": "Builds a greeting message for a user.",
  "responsibilities": ["format greeting"],
  "inputs": ["user"],
  "outputs": ["message"],
  "side_effects": [],
  "failure_modes": ["missing user"],
  "search_tags": ["greeting", "message", "user"]
}
""".strip()


class CapturingVectorStore:
    def __init__(self) -> None:
        self.points = []

    def upsert(self, points):
        self.points.extend(points)


def test_indexes_python_symbols_and_searches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        """
class Greeter:
    def greet(self, name: str) -> str:
        return f"hello {name}"

def build_message(user: str) -> str:
    return Greeter().greet(user)
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
    assert result["symbols"] == 3
    assert result["chunks"] == 4

    search = HybridSearch(indexer.sqlite_store, indexer.embedding_generator, repo_path=repo)
    hits = search.search("build_message", include_vectors=False)
    assert hits
    assert any(hit.symbol == "build_message" for hit in hits)


def test_qdrant_payloads_include_profiles_and_graph_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        """
RETRY_BACKOFF_MS = 250

def retry_delay(attempt: int) -> int:
    return RETRY_BACKOFF_MS * attempt
""".strip(),
        encoding="utf-8",
    )

    settings = Settings(sqlite_path=tmp_path / "codeatlas.db")
    vector_store = CapturingVectorStore()
    indexer = RepositoryIndexer(
        settings=settings,
        enable_qdrant=False,
        embedding_generator=HashEmbeddingGenerator(),
    )
    indexer.vector_store = vector_store
    indexer.index(repo)

    payloads = [payload for _, _, payload in vector_store.points]
    retry_payload = next(payload for payload in payloads if payload["symbol"] == "retry_delay")
    file_payload = next(payload for payload in payloads if payload["unit_type"] == "file_summary")

    assert retry_payload["language"] == "python"
    assert retry_payload["profile"]
    assert retry_payload["embedding_text"].startswith(retry_payload["profile"])
    assert retry_payload["graph_context"]
    assert retry_payload["graph_context"][0]["edge_type"] == "reads_config"
    assert file_payload["symbol_count"] == 2
    assert "retry_delay" in file_payload["defined_symbols"]


def test_graph_search_finds_config_metrics_and_impact(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        """
import logging

logger = logging.getLogger(__name__)
RETRY_BACKOFF_MS = 250
REQUEST_TIMEOUT_MS = 3000

class Metrics:
    def emit_latency(self, name: str, value: float) -> None:
        pass

metrics = Metrics()

def retry_delay(attempt: int) -> int:
    logger.info("retry configured")
    return RETRY_BACKOFF_MS * attempt

def load_user(user_id: str) -> dict:
    logger.info("loading user")
    metrics.emit_latency("user.load.latency", 12.5)
    return {"id": user_id, "timeout": REQUEST_TIMEOUT_MS}

def api_handler(request: dict) -> dict:
    return load_user(request["user_id"])
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

    assert result["edges"] > 0

    graph_search = GraphSearch(indexer.sqlite_store)
    retry_edges = graph_search.architecture_search("where is retry configured?")
    assert any(edge["edge_type"] == "reads_config" for edge in retry_edges)

    metric_edges = graph_search.architecture_search("which function emits latency metric?")
    assert any(edge["edge_type"] == "emits_metric" for edge in metric_edges)

    impact = graph_search.impact_analysis("load_user")
    assert impact is not None
    assert any(edge["edge_type"] == "calls" for edge in impact["edges"])

    config_impact = graph_search.impact_analysis("REQUEST_TIMEOUT_MS")
    assert config_impact is not None
    assert any(edge["edge_type"] == "reads_config" for edge in config_impact["edges"])


def test_optional_llm_enrichment_is_indexed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        """
def build_message(user: str) -> str:
    return f"hello {user}"
""".strip(),
        encoding="utf-8",
    )

    settings = Settings(sqlite_path=tmp_path / "codeatlas.db")
    indexer = RepositoryIndexer(
        settings=settings,
        enable_qdrant=False,
        embedding_generator=HashEmbeddingGenerator(),
        llm_enricher=LLMEnricher(FakeLLMClient()),
    )
    indexer.index(repo)

    hits = indexer.sqlite_store.search("missing user", limit=5)
    assert hits

    enrichment = indexer.sqlite_store.connection.execute(
        "SELECT description, tags, failure_modes FROM llm_enrichments"
    ).fetchone()
    assert enrichment["description"] == "Builds a greeting message for a user."
    assert "greeting" in enrichment["tags"]
    assert "missing user" in enrichment["failure_modes"]


def test_query_analyzer_boosts_exact_code_queries() -> None:
    exact_plan = QueryAnalyzer().classify("retryBackoffMs")
    semantic_plan = QueryAnalyzer().classify("where is retry logic configured")

    assert exact_plan.ripgrep_weight > exact_plan.fts_weight
    assert semantic_plan.vector_weight >= semantic_plan.ripgrep_weight


def test_ripgrep_retriever_returns_context_chunks(tmp_path: Path) -> None:
    if shutil.which("rg") is None:
        return

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        """
def handler():
    retryBackoffMs = 250
    return retryBackoffMs
""".strip(),
        encoding="utf-8",
    )

    chunks = RipgrepRetriever().search(repo, "retryBackoffMs", context_lines=1)

    assert chunks
    assert chunks[0].retrieval_method == "ripgrep"
    assert chunks[0].file_path == "service.py"
    assert "retryBackoffMs" in chunks[0].content


def test_hybrid_search_fuses_ripgrep_and_fts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        """
def handler():
    retryBackoffMs = 250
    return retryBackoffMs
""".strip(),
        encoding="utf-8",
    )

    settings = Settings(sqlite_path=tmp_path / "codeatlas.db")
    indexer = RepositoryIndexer(
        settings=settings,
        enable_qdrant=False,
        embedding_generator=HashEmbeddingGenerator(),
    )
    indexer.index(repo)

    search = HybridSearch(indexer.sqlite_store, indexer.embedding_generator, repo_path=repo)
    hits = search.search("retryBackoffMs", include_vectors=False)

    assert hits
    assert hits[0].retrieval_method in {"ripgrep", "fts"}
