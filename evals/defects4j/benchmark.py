from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from codeatlas.common.config import Settings
from codeatlas.embedding.factory import build_embedding_provider
from codeatlas.embedding.hash_provider import HashEmbeddingProvider
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.storage.sqlite_store import SQLiteStore
from evals.defects4j.run_fault_localization_eval import (
    _repo_tokens,
    _summary,
    evaluate_localization_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkout, index, and benchmark Defects4J localization cases.")
    parser.add_argument("--defects4j-home", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("evals/defects4j/benchmark_cases.yaml"))
    parser.add_argument("--work-dir", type=Path, default=Path("/private/tmp/codeatlas-defects4j-benchmark"))
    parser.add_argument("--output", type=Path, default=Path("evals/defects4j/results.json"))
    parser.add_argument("--config", type=Path, help="CodeAtlas config for embedding/Qdrant settings")
    parser.add_argument("--with-vectors", action="store_true", help="Index and query with Qdrant vectors")
    parser.add_argument("--reuse-checkouts", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--window-radius", type=int, default=12)
    args = parser.parse_args()

    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8")) or []
    grouped_cases = group_cases_by_bug(cases)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    bug_results: list[dict[str, Any]] = []
    for bug_id, bug_cases in grouped_cases.items():
        project, version = parse_bug_id(bug_id)
        checkout_dir = args.work_dir / "checkouts" / bug_id
        sqlite_path = args.work_dir / "indexes" / f"{bug_id}.db"
        checkout_defects4j_bug(
            args.defects4j_home,
            project,
            version,
            checkout_dir,
            reuse_checkout=args.reuse_checkouts,
        )
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        settings = settings_for_bug(args, sqlite_path, bug_id)
        embedding_provider = build_embedding_provider(settings.embeddings) if args.with_vectors else HashEmbeddingProvider()
        if not args.reuse_index:
            RepositoryIndexer(
                settings=settings,
                enable_qdrant=args.with_vectors,
                embedding_generator=embedding_provider,
            ).index(checkout_dir)

        sqlite_store = SQLiteStore(sqlite_path)
        raw_context_tokens = _repo_tokens(checkout_dir)
        case_results = [
            evaluate_localization_case(
                sqlite_store,
                embedding_provider,
                settings,
                checkout_dir,
                case,
                raw_context_tokens,
                limit=args.limit,
                include_vectors=args.with_vectors,
                window_radius=args.window_radius,
            )
            for case in bug_cases
        ]
        bug_results.append(
            {
                "bug_id": bug_id,
                "project": project,
                "version": version,
                "repo": str(checkout_dir),
                "sqlite_path": str(sqlite_path),
                "index_counts": index_counts(sqlite_store),
                "summary": _summary([case["metrics"] for case in case_results]),
                "cases": case_results,
            }
        )

    return {
        "cases_path": str(args.cases),
        "work_dir": str(args.work_dir),
        "with_vectors": args.with_vectors,
        "summary": _summary(
            [
                case["metrics"]
                for bug in bug_results
                for case in bug["cases"]
            ]
        ),
        "bugs": bug_results,
    }


def parse_bug_id(bug_id: str) -> tuple[str, str]:
    if "_" not in bug_id:
        raise ValueError(f"Expected bug_id like Lang_1b, got {bug_id}")
    project, version = bug_id.split("_", 1)
    if not project or not version:
        raise ValueError(f"Expected bug_id like Lang_1b, got {bug_id}")
    return project, version


def group_cases_by_bug(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["bug_id"])].append(case)
    return dict(grouped)


def checkout_defects4j_bug(
    defects4j_home: Path,
    project: str,
    version: str,
    checkout_dir: Path,
    reuse_checkout: bool,
) -> None:
    if reuse_checkout and checkout_dir.exists():
        return
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(defects4j_home / "framework/bin/defects4j"),
        "checkout",
        "-p",
        project,
        "-v",
        version,
        "-w",
        str(checkout_dir),
    ]
    subprocess.run(command, cwd=defects4j_home, check=True)


def settings_for_bug(args: argparse.Namespace, sqlite_path: Path, bug_id: str) -> Settings:
    base = Settings.from_yaml(args.config) if args.config else Settings(sqlite_path=sqlite_path)
    qdrant_collection = base.qdrant_collection
    if args.with_vectors:
        qdrant_collection = f"{base.qdrant_collection}_{bug_id.lower()}"
    return Settings.from_dict(
        {
            "sqlite_path": sqlite_path,
            "qdrant_url": base.qdrant_url,
            "qdrant_collection": qdrant_collection,
            "embeddings": {
                "provider": base.embeddings.provider,
                "model": base.embeddings.model,
                "dimensions": base.embeddings.dimensions,
                "batch_size": base.embeddings.batch_size,
            },
            "llm": {
                "enabled": base.llm.enabled,
                "provider": base.llm.provider,
                "model": base.llm.model,
                "temperature": base.llm.temperature,
                "max_tokens": base.llm.max_tokens,
                "base_url": base.llm.base_url,
                "api_key": base.llm.api_key,
            },
        }
    )


def index_counts(sqlite_store: SQLiteStore) -> dict[str, int]:
    connection = sqlite_store.connection
    return {
        "files": int(connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]),
        "symbols": int(connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]),
        "chunks": int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
        "edges": int(connection.execute("SELECT COUNT(*) FROM symbol_edges").fetchone()[0]),
    }


if __name__ == "__main__":
    main()
