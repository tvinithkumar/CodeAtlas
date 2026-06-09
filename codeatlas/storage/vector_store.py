from __future__ import annotations

from codeatlas.common.config import Settings


class QdrantVectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.settings.qdrant_url)
        return self._client

    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        if any(item.name == self.settings.qdrant_collection for item in collections):
            return
        self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=VectorParams(size=self.settings.embeddings.dimensions, distance=Distance.COSINE),
        )

    def check_available(self) -> None:
        self.client.get_collections()

    def upsert(self, points: list[tuple[str, list[float], dict[str, object]]], batch_size: int = 128) -> None:
        if not points:
            return
        from qdrant_client.models import PointStruct

        self.ensure_collection()
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=[PointStruct(id=point_id, vector=vector, payload=payload) for point_id, vector, payload in batch],
            )

    def search(self, vector: list[float], limit: int = 10) -> list[dict[str, object]]:
        self.ensure_collection()
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                limit=limit,
            )
            hits = result.points
        else:
            hits = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=vector,
                limit=limit,
            )
        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "file_path": hit.payload.get("file_path", ""),
                "symbol": hit.payload.get("symbol", ""),
                "line_start": hit.payload.get("line_start"),
                "line_end": hit.payload.get("line_end"),
                "unit_type": hit.payload.get("unit_type", ""),
                "language": hit.payload.get("language", ""),
                "profile": hit.payload.get("profile", ""),
                "embedding_text": hit.payload.get("embedding_text", ""),
                "graph_context": hit.payload.get("graph_context", []),
                "symbol_count": hit.payload.get("symbol_count"),
                "defined_symbols": hit.payload.get("defined_symbols", []),
                "content": hit.payload.get("content", ""),
            }
            for hit in hits
        ]
