from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from codeatlas.common.config import Settings
from codeatlas.enrichment.embedding_generator import HashEmbeddingGenerator
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.retrieval.fts_search import FTSSearch
from codeatlas.retrieval.hybrid_search import HybridSearch
from codeatlas.retrieval.ripgrep_search import RipgrepRetriever
from codeatlas.storage.models import SearchHit
from codeatlas.storage.sqlite_store import SQLiteStore
from evals.metrics import CaseMetrics, evaluate_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CodeAtlas retrieval quality evals.")
    parser.add_argument("--repo", type=Path, default=Path("examples/benchmark_repo"))
    parser.add_argument("--cases", type=Path, default=Path("evals/benchmark_cases.yaml"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    with TemporaryDirectory() as tmpdir:
        settings = Settings(sqlite_path=Path(tmpdir) / "codeatlas-eval.db")
        indexer = RepositoryIndexer(
            settings=settings,
            enable_qdrant=False,
            embedding_generator=HashEmbeddingGenerator(),
        )
        indexer.index(args.repo)
        sqlite_store = indexer.sqlite_store

        cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
        raw_context_tokens = _repo_tokens(args.repo)

        results = {
            "ripgrep": _evaluate_ripgrep(args.repo, cases, raw_context_tokens, args.limit),
            "fts": _evaluate_fts(sqlite_store, cases, raw_context_tokens, args.limit),
            "hybrid": _evaluate_hybrid(sqlite_store, indexer.embedding_generator, args.repo, cases, raw_context_tokens, args.limit),
        }
        print(json.dumps(results, indent=2))


def _evaluate_ripgrep(
    repo: Path,
    cases: list[dict[str, object]],
    raw_context_tokens: int,
    limit: int,
) -> dict[str, object]:
    retriever = RipgrepRetriever()
    return _evaluate_mode(
        cases,
        lambda query: retriever.search_hits(repo, query, max_results=limit),
        raw_context_tokens,
    )


def _evaluate_fts(
    sqlite_store: SQLiteStore,
    cases: list[dict[str, object]],
    raw_context_tokens: int,
    limit: int,
) -> dict[str, object]:
    retriever = FTSSearch(sqlite_store)
    return _evaluate_mode(cases, lambda query: retriever.search(query, limit=limit), raw_context_tokens)


def _evaluate_hybrid(
    sqlite_store: SQLiteStore,
    embedding_generator: HashEmbeddingGenerator,
    repo: Path,
    cases: list[dict[str, object]],
    raw_context_tokens: int,
    limit: int,
) -> dict[str, object]:
    retriever = HybridSearch(sqlite_store, embedding_generator, repo_path=repo)
    return _evaluate_mode(
        cases,
        lambda query: retriever.search(query, limit=limit, include_vectors=False),
        raw_context_tokens,
    )


def _evaluate_mode(
    cases: list[dict[str, object]],
    search_fn,
    raw_context_tokens: int,
) -> dict[str, object]:
    case_results: list[dict[str, object]] = []
    metrics: list[CaseMetrics] = []
    for case in cases:
        hits = search_fn(str(case["query"]))
        case_metrics = evaluate_case(
            hits,
            expected_symbols=list(case.get("expected_symbols", [])),
            expected_files=list(case.get("expected_files", [])),
            raw_context_tokens=raw_context_tokens,
        )
        metrics.append(case_metrics)
        case_results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "metrics": asdict(case_metrics),
                "hits": [_hit_summary(hit) for hit in hits[:5]],
            }
        )

    return {"summary": _summary(metrics), "cases": case_results}


def _summary(metrics: list[CaseMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].__dataclass_fields__.keys()
    return {
        key: sum(getattr(item, key) for item in metrics) / len(metrics)
        for key in keys
    }


def _hit_summary(hit: SearchHit) -> dict[str, object]:
    return {
        "symbol": hit.symbol,
        "file_path": hit.file_path,
        "retrieval_method": hit.retrieval_method,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
    }


def _repo_tokens(repo: Path) -> int:
    total = 0
    for path in repo.rglob("*"):
        if path.is_file():
            total += len(path.read_text(encoding="utf-8", errors="replace").split())
    return total


if __name__ == "__main__":
    main()

