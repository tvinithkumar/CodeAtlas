from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.enrichment.embedding_generator import LocalEmbeddingGenerator
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.retrieval.code_window import CodeWindowFetcher
from codeatlas.retrieval.graph_search import GraphSearch
from codeatlas.retrieval.hybrid_search import HybridSearch
from codeatlas.storage.sqlite_store import SQLiteStore
from codeatlas.storage.vector_store import QdrantVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="codeatlas")
    parser.add_argument("--sqlite-path", type=Path, default=Settings().sqlite_path)
    parser.add_argument("--config", type=Path, help="YAML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("repo", type=Path)
    index_parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant vector upserts")
    index_parser.add_argument("--with-llm", action="store_true", help="Enable optional offline LLM enrichment")

    search_parser = subparsers.add_parser("search", help="Search indexed code")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--no-vectors", action="store_true", help="Use SQLite FTS only")
    search_parser.add_argument("--repo", type=Path, help="Repository path for ripgrep retrieval")

    explain_parser = subparsers.add_parser("explain", help="Explain a symbol and its graph edges")
    explain_parser.add_argument("symbol")

    usage_parser = subparsers.add_parser("find-usage", help="Find incoming references to a symbol")
    usage_parser.add_argument("symbol")

    impact_parser = subparsers.add_parser("impact", help="Trace incoming impact for a symbol")
    impact_parser.add_argument("symbol")
    impact_parser.add_argument("--depth", type=int, default=2)

    window_parser = subparsers.add_parser("window", help="Fetch a bounded code window")
    window_parser.add_argument("--repo", type=Path, required=True)
    window_parser.add_argument("--file", required=True)
    window_parser.add_argument("--line", type=int, required=True)
    window_parser.add_argument("--radius", type=int, default=20)

    args = parser.parse_args()
    settings = Settings.from_yaml(args.config) if args.config else Settings(sqlite_path=args.sqlite_path)

    if args.command == "index":
        if args.with_llm and not settings.llm.enabled:
            settings = Settings.from_dict(
                {
                    "sqlite_path": settings.sqlite_path,
                    "qdrant_url": settings.qdrant_url,
                    "qdrant_collection": settings.qdrant_collection,
                    "embedding_dimension": settings.embedding_dimension,
                    "embedding_model": settings.embedding_model,
                    "llm": {
                        "enabled": True,
                        "provider": settings.llm.provider,
                        "model": settings.llm.model,
                        "temperature": settings.llm.temperature,
                    },
                }
            )
        result = RepositoryIndexer(settings=settings, enable_qdrant=not args.no_qdrant).index(args.repo)
        print(json.dumps(result, indent=2))
        return

    if args.command == "window":
        window = CodeWindowFetcher().get_code_window(args.repo, args.file, args.line, args.radius)
        print(json.dumps(window.__dict__, indent=2))
        return

    sqlite_store = SQLiteStore(settings.sqlite_path)
    graph_search = GraphSearch(sqlite_store)

    if args.command == "search":
        vector_store = None if args.no_vectors else QdrantVectorStore(settings)
        search = HybridSearch(
            sqlite_store,
            LocalEmbeddingGenerator(settings.embedding_model),
            vector_store,
            repo_path=args.repo,
        )
        hits = [hit.__dict__ for hit in search.search(args.query, limit=args.limit, include_vectors=not args.no_vectors)]
        if not hits:
            hits = graph_search.architecture_search(args.query, limit=args.limit)
        print(json.dumps(hits, indent=2))
        return

    if args.command == "explain":
        print(json.dumps(graph_search.explain_symbol(args.symbol), indent=2))
        return

    if args.command == "find-usage":
        print(json.dumps(graph_search.find_usage(args.symbol), indent=2))
        return

    if args.command == "impact":
        print(json.dumps(graph_search.impact_analysis(args.symbol, depth=args.depth), indent=2))


if __name__ == "__main__":
    main()
