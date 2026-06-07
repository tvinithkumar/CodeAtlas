from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbeddingGenerator(Protocol):
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalEmbeddingGenerator:
    """Local embedding generator backed by FastEmbed.

    The default model is BAAI/bge-small-en-v1.5. Model files are downloaded by
    FastEmbed the first time this is used.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return list(next(self.model.embed([text])))


class HashEmbeddingGenerator:
    """Deterministic test embedding generator."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
