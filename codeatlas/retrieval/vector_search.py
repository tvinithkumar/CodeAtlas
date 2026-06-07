from __future__ import annotations

from codeatlas.enrichment.embedding_generator import EmbeddingGenerator
from codeatlas.storage.models import SearchHit
from codeatlas.storage.vector_store import QdrantVectorStore


class VectorSearch:
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: QdrantVectorStore) -> None:
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        vector = self.embedding_generator.embed(query)
        return [
            SearchHit(
                id=str(row["id"]),
                score=float(row["score"]),
                file_path=str(row["file_path"]),
                symbol=str(row["symbol"]),
                content=str(row["content"]),
                source="qdrant",
                retrieval_method="vector",
            )
            for row in self.vector_store.search(vector, limit=limit)
        ]

