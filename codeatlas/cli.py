from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.embedding.factory import build_embedding_provider
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
    index_parser.add_argument("--llm-enabled", action="store_true", help="Enable optional offline LLM enrichment")
    index_parser.add_argument("--llm-provider", choices=["ollama", "lmstudio"], help="LLM provider")
    index_parser.add_argument("--llm-model", help="LLM model name")
    index_parser.add_argument("--llm-base-url", help="OpenAI-compatible LLM base URL")
    index_parser.add_argument("--llm-api-key", help="OpenAI-compatible LLM API key")
    index_parser.add_argument("--llm-temperature", type=float, help="LLM temperature")
    index_parser.add_argument("--llm-max-tokens", type=int, help="LLM max output tokens")
    index_parser.add_argument("--embedding-provider", choices=["fastembed", "sentence_transformers", "hash"])
    index_parser.add_argument("--embedding-model")
    index_parser.add_argument("--embedding-dimensions", type=int)
    index_parser.add_argument("--embedding-batch-size", type=int)

    search_parser = subparsers.add_parser("search", help="Search indexed code")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--no-vectors", action="store_true", help="Use SQLite FTS only")
    search_parser.add_argument("--repo", type=Path, help="Repository path for ripgrep retrieval")

    explain_parser = subparsers.add_parser("explain", help="Explain a symbol and its graph edges")
    explain_parser.add_argument("symbol")

    usage_parser = subparsers.add_parser("find-usage", help="Find incoming references to a symbol")
    usage_parser.add_argument("symbol")

    related_parser = subparsers.add_parser("related", help="Find symbols related by graph edges")
    related_parser.add_argument("symbol")
    related_parser.add_argument("--limit", type=int, default=10)
    related_parser.add_argument("--repo", type=Path, help="Accepted for CLI symmetry; graph search uses the index")

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
        settings = _settings_with_llm_overrides(settings, args)
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
            build_embedding_provider(settings.embeddings),
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

    if args.command == "related":
        print(json.dumps(graph_search.related_symbols(args.symbol, limit=args.limit), indent=2))
        return

    if args.command == "impact":
        print(json.dumps(graph_search.impact_analysis(args.symbol, depth=args.depth), indent=2))


def _settings_with_llm_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    enabled = settings.llm.enabled or args.with_llm or args.llm_enabled
    provider = args.llm_provider or settings.llm.provider
    model = args.llm_model or settings.llm.model
    temperature = args.llm_temperature if args.llm_temperature is not None else settings.llm.temperature
    max_tokens = args.llm_max_tokens if args.llm_max_tokens is not None else settings.llm.max_tokens
    base_url = args.llm_base_url or settings.llm.base_url
    api_key = args.llm_api_key or settings.llm.api_key
    embedding_provider = args.embedding_provider or settings.embeddings.provider
    embedding_model = args.embedding_model or settings.embeddings.model
    embedding_dimensions = (
        args.embedding_dimensions if args.embedding_dimensions is not None else settings.embeddings.dimensions
    )
    embedding_batch_size = (
        args.embedding_batch_size if args.embedding_batch_size is not None else settings.embeddings.batch_size
    )

    return Settings.from_dict(
        {
            "sqlite_path": settings.sqlite_path,
            "qdrant_url": settings.qdrant_url,
            "qdrant_collection": settings.qdrant_collection,
            "embeddings": {
                "provider": embedding_provider,
                "model": embedding_model,
                "dimensions": embedding_dimensions,
                "batch_size": embedding_batch_size,
            },
            "llm": {
                "enabled": enabled,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "base_url": base_url,
                "api_key": api_key,
            },
        }
    )


if __name__ == "__main__":
    main()
