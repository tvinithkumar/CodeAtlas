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

    def upsert(self, points: list[tuple[str, list[float], dict[str, object]]]) -> None:
        if not points:
            return
        from qdrant_client.models import PointStruct

        self.ensure_collection()
        self.client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload) for point_id, vector, payload in points],
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
                "content": hit.payload.get("content", ""),
            }
            for hit in hits
        ]
