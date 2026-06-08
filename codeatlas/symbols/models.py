from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    qualified_name: str
    parent: str | None = None

    @property
    def id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.file_path}:{self.qualified_name}"))


@dataclass(frozen=True)
class Relationship:
    source: str
    target: str
    kind: str
    file_path: str
    confidence: float = 1.0
    line_start: int | None = None
    line_end: int | None = None
