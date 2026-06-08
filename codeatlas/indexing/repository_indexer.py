from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from codeatlas.chunking.ast_chunker import ASTChunker
from codeatlas.chunking.models import CodeChunk
from codeatlas.common.config import Settings
from codeatlas.embedding.base import EmbeddingProvider
from codeatlas.embedding.factory import build_embedding_provider
from codeatlas.enrichment.llm.enricher import LLMEnricher
from codeatlas.enrichment.llm.lmstudio_client import LMStudioClient
from codeatlas.enrichment.llm.litellm_client import LiteLLMClient
from codeatlas.enrichment.symbol_profiler import SymbolProfiler
from codeatlas.ingestion.repository_loader import RepositoryLoader
from codeatlas.storage.sqlite_store import SQLiteStore
from codeatlas.storage.vector_store import QdrantVectorStore
from codeatlas.symbols.extractor import SymbolExtractor
from codeatlas.symbols.models import Relationship, Symbol
from codeatlas.symbols.relationships import RelationshipExtractor


class RepositoryIndexer:
    def __init__(
        self,
        settings: Settings | None = None,
        enable_qdrant: bool = True,
        embedding_generator: EmbeddingProvider | None = None,
        llm_enricher: LLMEnricher | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.loader = RepositoryLoader(self.settings)
        self.extractor = SymbolExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.chunker = ASTChunker()
        self.profiler = SymbolProfiler()
        self.llm_enricher = llm_enricher or self._build_llm_enricher()
        self.embedding_generator = embedding_generator or build_embedding_provider(self.settings.embeddings)
        self.sqlite_store = SQLiteStore(self.settings.sqlite_path)
        self.vector_store = QdrantVectorStore(self.settings) if enable_qdrant else None

    def index(self, repo_path: Path) -> dict[str, int]:
        files = self.loader.load(repo_path)
        symbol_count = 0
        chunk_count = 0
        vector_points: list[tuple[str, list[float], dict[str, object]]] = []

        for source_file in files:
            self.sqlite_store.upsert_file(source_file.relative_path, source_file.language, source_file.content)
            symbols = self.extractor.extract(source_file)
            chunks = self.chunker.chunk(source_file, symbols)
            file_summary = self._file_summary_chunk(source_file.relative_path, source_file.language, symbols)
            chunks.append(file_summary)
            chunks_by_symbol = {chunk.symbol_qualified_name: chunk for chunk in chunks}
            relationships = self._resolve_relationships(self.relationship_extractor.extract(source_file, symbols), symbols)

            for symbol in symbols:
                self.sqlite_store.upsert_symbol(symbol)
                symbol_count += 1
                chunk = chunks_by_symbol.get(symbol.qualified_name)
                if chunk is None:
                    continue
                profile = self._profile_symbol(symbol, chunk)
                self.sqlite_store.upsert_chunk(chunk, profile)
                chunk_count += 1
                if self.vector_store is not None:
                    vector_points.append(
                        (
                            chunk.id,
                            self.embedding_generator.embed(f"{profile}\n{chunk.content}"),
                            {
                                "unit_type": chunk.unit_type,
                                "file_path": chunk.file_path,
                                "symbol": chunk.symbol_qualified_name,
                                "line_start": chunk.start_line,
                                "line_end": chunk.end_line,
                                "content": chunk.content,
                            },
                        )
                    )

            file_profile = self._file_profile(source_file.relative_path, symbols)
            self.sqlite_store.upsert_chunk(file_summary, file_profile)
            chunk_count += 1
            if self.vector_store is not None:
                vector_points.append(
                    (
                        file_summary.id,
                        self.embedding_generator.embed(f"{file_profile}\n{file_summary.content}"),
                        {
                            "unit_type": file_summary.unit_type,
                            "file_path": file_summary.file_path,
                            "symbol": file_summary.symbol_qualified_name,
                            "line_start": file_summary.start_line,
                            "line_end": file_summary.end_line,
                            "content": file_summary.content,
                        },
                    )
                )
            self.sqlite_store.replace_edges(source_file.relative_path, relationships)

        self.sqlite_store.commit()
        if self.vector_store is not None:
            try:
                self.vector_store.upsert(vector_points)
            except Exception:
                pass

        edge_count = sum(
            len(self.sqlite_store.get_symbol_edges(symbol.id, direction="out"))
            for source_file in files
            for symbol in self.extractor.extract(source_file)
        )
        return {"files": len(files), "symbols": symbol_count, "chunks": chunk_count, "edges": edge_count}

    def _build_llm_enricher(self) -> LLMEnricher | None:
        if not self.settings.llm.enabled:
            return None
        if self.settings.llm.provider == "lmstudio":
            return LLMEnricher(LMStudioClient(self.settings.llm))
        return LLMEnricher(LiteLLMClient(self.settings.llm))

    def _profile_symbol(self, symbol: Symbol, chunk: CodeChunk) -> str:
        base_profile = self.profiler.describe(symbol, chunk)
        if self.llm_enricher is None:
            return base_profile
        try:
            enrichment = self.llm_enricher.enrich(symbol, chunk)
        except Exception:
            return base_profile
        if enrichment.is_empty:
            return base_profile
        self.sqlite_store.upsert_llm_enrichment(chunk.id, enrichment)
        return f"{base_profile}\n{enrichment.to_profile_text()}"

    def _resolve_relationships(self, relationships: list[Relationship], symbols: list[Symbol]) -> list[Relationship]:
        symbol_ids_by_name = {symbol.name: symbol.id for symbol in symbols}
        symbol_ids_by_qualified = {symbol.qualified_name: symbol.id for symbol in symbols}
        resolved: list[Relationship] = []
        for edge in relationships:
            source = symbol_ids_by_qualified.get(edge.source, edge.source)
            target = (
                symbol_ids_by_qualified.get(edge.target)
                or symbol_ids_by_name.get(edge.target)
                or edge.target
            )
            resolved.append(Relationship(source, target, edge.kind, edge.file_path, edge.confidence))
        return resolved

    def _file_summary_chunk(self, relative_path: str, language: str, symbols: list[Symbol]) -> CodeChunk:
        content = self._file_profile(relative_path, symbols)
        return CodeChunk(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{relative_path}:file_summary")),
            unit_type="file_summary",
            symbol_qualified_name=relative_path,
            file_path=relative_path,
            language=language,
            start_line=1,
            end_line=1,
            content=content,
        )

    def _file_profile(self, relative_path: str, symbols: list[Symbol]) -> str:
        names = ", ".join(symbol.qualified_name for symbol in symbols) or "no symbols"
        return f"File {relative_path} defines {names}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a repository with the CodeAtlas MVP pipeline.")
    parser.add_argument("repo", type=Path, help="Repository path to index")
    parser.add_argument("--sqlite-path", type=Path, default=Settings().sqlite_path)
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant vector upserts")
    parser.add_argument("--with-llm", action="store_true", help="Enable optional offline LLM enrichment")
    parser.add_argument("--llm-enabled", action="store_true", help="Enable optional offline LLM enrichment")
    parser.add_argument("--llm-provider", choices=["ollama", "lmstudio"], help="LLM provider")
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible LLM base URL")
    parser.add_argument("--llm-api-key", help="OpenAI-compatible LLM API key")
    parser.add_argument("--llm-temperature", type=float, help="LLM temperature")
    parser.add_argument("--llm-max-tokens", type=int, help="LLM max output tokens")
    parser.add_argument("--embedding-provider", choices=["fastembed", "sentence_transformers", "hash"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--embedding-batch-size", type=int)
    args = parser.parse_args()

    settings = Settings.from_yaml(args.config) if args.config else Settings(sqlite_path=args.sqlite_path)
    settings = _settings_with_llm_overrides(settings, args)
    result = RepositoryIndexer(settings=settings, enable_qdrant=not args.no_qdrant).index(args.repo)
    print(result)


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
