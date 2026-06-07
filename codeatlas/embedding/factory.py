from __future__ import annotations

from codeatlas.common.config import EmbeddingSettings
from codeatlas.embedding.base import EmbeddingProvider
from codeatlas.embedding.fastembed_provider import FastEmbedProvider
from codeatlas.embedding.hash_provider import HashEmbeddingProvider
from codeatlas.embedding.sentence_transformers_provider import SentenceTransformersProvider


def build_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    if settings.provider == "hash":
        return HashEmbeddingProvider(settings.dimensions)
    if settings.provider == "sentence_transformers":
        return SentenceTransformersProvider(settings.model, batch_size=settings.batch_size)
    if settings.provider == "fastembed":
        return FastEmbedProvider(settings.model)
    raise ValueError(f"Unsupported embedding provider: {settings.provider}")
