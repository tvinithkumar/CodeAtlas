from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    file_path: str
    symbol: str
    content: str
    source: str = "sqlite"
    retrieval_method: str = "fts"
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class RetrievalChunk:
    chunk_id: str
    source: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    retrieval_method: str
