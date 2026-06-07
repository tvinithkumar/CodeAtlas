"""Embedding providers for CodeAtlas."""

from codeatlas.embedding.base import EmbeddingProvider
from codeatlas.embedding.factory import build_embedding_provider
from codeatlas.embedding.fastembed_provider import FastEmbedProvider
from codeatlas.embedding.hash_provider import HashEmbeddingProvider
from codeatlas.embedding.sentence_transformers_provider import SentenceTransformersProvider

__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
    "HashEmbeddingProvider",
    "SentenceTransformersProvider",
    "build_embedding_provider",
]

