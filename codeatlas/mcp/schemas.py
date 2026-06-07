from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCPToolConfig:
    repo_path: str
    sqlite_path: str
    max_results: int
    max_lines_per_result: int
    max_total_chars: int

