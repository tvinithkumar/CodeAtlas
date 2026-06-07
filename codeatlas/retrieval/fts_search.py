from __future__ import annotations

from codeatlas.storage.models import SearchHit
from codeatlas.storage.sqlite_store import SQLiteStore


class FTSSearch:
    def __init__(self, sqlite_store: SQLiteStore) -> None:
        self.sqlite_store = sqlite_store

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        return [
            SearchHit(
                id=str(row["id"]),
                score=1.0,
                file_path=str(row["file_path"]),
                symbol=str(row["symbol_qualified_name"]),
                content=str(row["content"]),
                source="sqlite",
                retrieval_method="fts",
            )
            for row in self.sqlite_store.search(query, limit=limit)
        ]

