from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    id: str
    unit_type: str
    symbol_qualified_name: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    content: str
