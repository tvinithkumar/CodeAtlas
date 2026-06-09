from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from codeatlas.common.config import Settings
from codeatlas.embedding.factory import build_embedding_provider
from codeatlas.embedding.hash_provider import HashEmbeddingProvider
from codeatlas.graph.impact_analyzer import ImpactAnalyzer
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.retrieval.hybrid_search import HybridSearch
from codeatlas.storage.models import SearchHit
from codeatlas.storage.sqlite_store import SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Defects4J fault localization evals.")
    parser.add_argument("--repo", type=Path, required=True, help="Checked-out Defects4J bug version")
    parser.add_argument("--cases", type=Path, default=Path("evals/defects4j/benchmark_cases.yaml"))
    parser.add_argument("--sqlite-path", type=Path, help="SQLite DB path. Uses a temp DB by default.")
    parser.add_argument("--config", type=Path, help="CodeAtlas config for embeddings/Qdrant settings")
    parser.add_argument("--reuse-index", action="store_true", help="Use an existing SQLite index")
    parser.add_argument("--with-vectors", action="store_true", help="Include Qdrant vector retrieval")
    parser.add_argument("--allow-vector-errors", action="store_true", help="Continue if Qdrant vector retrieval fails")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--window-radius", type=int, default=12)
    args = parser.parse_args()

    try:
        if args.sqlite_path is None:
            with TemporaryDirectory() as tmpdir:
                sqlite_path = Path(tmpdir) / "defects4j-localization.db"
                result = run_eval(args, sqlite_path)
                print(json.dumps(result, indent=2))
            return

        result = run_eval(args, args.sqlite_path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))


def run_eval(args: argparse.Namespace, sqlite_path: Path) -> dict[str, Any]:
    settings = Settings.from_yaml(args.config) if args.config else Settings(sqlite_path=sqlite_path)
    if args.config:
        settings = Settings.from_dict(
            {
                "sqlite_path": sqlite_path,
                "qdrant_url": settings.qdrant_url,
                "qdrant_collection": settings.qdrant_collection,
                "embeddings": {
                    "provider": settings.embeddings.provider,
                    "model": settings.embeddings.model,
                    "dimensions": settings.embeddings.dimensions,
                    "batch_size": settings.embeddings.batch_size,
                },
                "llm": {
                    "enabled": settings.llm.enabled,
                    "provider": settings.llm.provider,
                    "model": settings.llm.model,
                    "temperature": settings.llm.temperature,
                    "max_tokens": settings.llm.max_tokens,
                    "base_url": settings.llm.base_url,
                    "api_key": settings.llm.api_key,
                },
            }
        )

    embedding_provider = build_embedding_provider(settings.embeddings) if args.with_vectors else HashEmbeddingProvider()
    if args.with_vectors and not args.allow_vector_errors:
        check_qdrant_available(settings)
    if not args.reuse_index:
        RepositoryIndexer(
            settings=settings,
            enable_qdrant=args.with_vectors,
            embedding_generator=embedding_provider,
        ).index(args.repo)

    sqlite_store = SQLiteStore(sqlite_path)
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8")) or []
    raw_context_tokens = _repo_tokens(args.repo)
    case_results = [
        evaluate_localization_case(
            sqlite_store,
            embedding_provider,
            settings,
            args.repo,
            case,
            raw_context_tokens,
            limit=args.limit,
            include_vectors=args.with_vectors,
            window_radius=args.window_radius,
        )
        for case in cases
    ]

    return {
        "repo": str(args.repo),
        "cases_path": str(args.cases),
        "summary": _summary([case["metrics"] for case in case_results]),
        "cases": case_results,
    }


def evaluate_localization_case(
    sqlite_store: SQLiteStore,
    embedding_provider,
    settings: Settings,
    repo: Path,
    case: dict[str, Any],
    raw_context_tokens: int,
    limit: int = 10,
    include_vectors: bool = False,
    window_radius: int = 12,
) -> dict[str, Any]:
    vector_store = None
    if include_vectors:
        from codeatlas.storage.vector_store import QdrantVectorStore

        vector_store = QdrantVectorStore(settings)

    search = HybridSearch(sqlite_store, embedding_provider, vector_store=vector_store, repo_path=repo)
    search_result = search.search_detailed(str(case["query"]), limit=limit, include_vectors=include_vectors)
    hits = search_result.hits
    impact_symbol = str(case.get("impact_symbol") or _first_expected_method(case) or case["query"])
    impact = ImpactAnalyzer(sqlite_store).analyze(
        impact_symbol,
        repo_path=repo,
        depth=2,
        limit=limit,
        window_radius=window_radius,
    )

    ranked_files = _rank_files(hits, impact)
    ranked_methods = _rank_methods(hits, impact)
    retrieved_tokens = _retrieved_tokens(hits, impact)
    expected_files = list(case.get("expected_files", []))
    expected_methods = list(case.get("expected_methods", []))
    metrics = {
        "file_recall_at_1": recall_at_k(ranked_files, expected_files, 1),
        "file_recall_at_5": recall_at_k(ranked_files, expected_files, 5),
        "file_recall_at_10": recall_at_k(ranked_files, expected_files, 10),
        "method_recall_at_1": recall_at_k(ranked_methods, expected_methods, 1) if expected_methods else None,
        "method_recall_at_5": recall_at_k(ranked_methods, expected_methods, 5) if expected_methods else None,
        "method_recall_at_10": recall_at_k(ranked_methods, expected_methods, 10) if expected_methods else None,
        "mrr": reciprocal_rank([*ranked_methods, *ranked_files], _relevant_items(case)),
        "context_compression_ratio": compression_ratio(raw_context_tokens, retrieved_tokens),
    }

    return {
        "bug_id": case["bug_id"],
        "query": case["query"],
        "impact_symbol": impact_symbol,
        "metrics": metrics,
        "retrieval_method_counts": _retrieval_method_counts(hits),
        "raw_retrieval_method_counts": search_result.raw_retrieval_method_counts,
        "vector_hit_count": search_result.raw_retrieval_method_counts.get("vector", 0),
        "fused_vector_hit_count": sum(1 for hit in hits if hit.retrieval_method == "vector"),
        "retrieval_errors": search_result.errors,
        "ranked_files": ranked_files[:5],
        "ranked_methods": ranked_methods[:10],
        "search_hits": [_hit_summary(hit) for hit in hits[:5]],
        "impact": _impact_summary(impact),
    }


def check_qdrant_available(settings: Settings) -> None:
    from codeatlas.storage.vector_store import QdrantVectorStore

    try:
        QdrantVectorStore(settings).check_available()
    except Exception as exc:
        raise RuntimeError(
            "Qdrant is not reachable for a vector benchmark. Start Qdrant on "
            f"{settings.qdrant_url}, rerun without --with-vectors, or pass "
            "--allow-vector-errors to record fallback metrics."
        ) from exc


def recall_at_k(ranked_items: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    top = ranked_items[:k]
    found = {item for item in expected if any(_matches(candidate, item) for candidate in top)}
    return len(found) / len(set(expected))


def reciprocal_rank(ranked_items: list[str], expected: list[str]) -> float:
    for index, item in enumerate(ranked_items, start=1):
        if any(_matches(item, expected_item) for expected_item in expected):
            return 1.0 / index
    return 0.0


def compression_ratio(raw_context_tokens: int, retrieved_tokens: int) -> float:
    if retrieved_tokens <= 0:
        return 0.0
    return raw_context_tokens / retrieved_tokens


def _rank_files(hits: list[SearchHit], impact: dict[str, Any] | None) -> list[str]:
    ranked: list[str] = []
    for hit in hits:
        _append_unique(ranked, hit.file_path)
    if impact:
        for item in impact.get("files_affected", []):
            _append_unique(ranked, str(item["file_path"]))
        for window in impact.get("top_code_windows", []):
            _append_unique(ranked, str(window["file_path"]))
    return ranked


def _rank_methods(hits: list[SearchHit], impact: dict[str, Any] | None) -> list[str]:
    ranked: list[str] = []
    if impact:
        _append_unique(ranked, str(impact["symbol"]["qualified_name"]))
        for edge in impact.get("direct_callers", []):
            _append_unique(ranked, str(edge.get("src_symbol") or edge["src_symbol_id"]))
    for hit in hits:
        _append_unique(ranked, hit.symbol)
    if impact:
        for item in impact.get("related_methods", []):
            _append_unique(ranked, str(item["symbol"]))
    return ranked


def _retrieved_tokens(hits: list[SearchHit], impact: dict[str, Any] | None) -> int:
    total = sum(len(hit.content.split()) for hit in hits)
    if impact:
        total += sum(len(window["content"].split()) for window in impact.get("top_code_windows", []))
    return total


def _impact_summary(impact: dict[str, Any] | None) -> dict[str, Any] | None:
    if impact is None:
        return None
    return {
        "symbol": impact["symbol"],
        "direct_callers": impact["direct_callers"][:5],
        "files_affected": impact["files_affected"][:5],
        "risk_notes": impact["risk_notes"],
        "top_code_windows": [
            {
                "file_path": window["file_path"],
                "line_start": window["line_start"],
                "line_end": window["line_end"],
                "reason": window["reason"],
            }
            for window in impact.get("top_code_windows", [])[:5]
        ],
    }


def _summary(metrics: list[dict[str, float | None]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted({key for item in metrics for key in item})
    summary: dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in metrics if item.get(key) is not None]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def _hit_summary(hit: SearchHit) -> dict[str, Any]:
    return {
        "symbol": hit.symbol,
        "file_path": hit.file_path,
        "retrieval_method": hit.retrieval_method,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
    }


def _retrieval_method_counts(hits: list[SearchHit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.retrieval_method] = counts.get(hit.retrieval_method, 0) + 1
    return counts


def _repo_tokens(repo: Path) -> int:
    total = 0
    for path in repo.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            total += len(path.read_text(encoding="utf-8", errors="replace").split())
    return total


def _first_expected_method(case: dict[str, Any]) -> str | None:
    methods = list(case.get("expected_methods", []))
    if not methods:
        return None
    return str(methods[0]).split(".")[-1]


def _relevant_items(case: dict[str, Any]) -> list[str]:
    return [*list(case.get("expected_methods", [])), *list(case.get("expected_files", []))]


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _matches(candidate: str, expected: str) -> bool:
    return candidate == expected or candidate.endswith(expected) or expected.endswith(candidate)


if __name__ == "__main__":
    main()
