from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeWindow:
    file_path: str
    line_start: int
    line_end: int
    content: str


class CodeWindowFetcher:
    def get_code_window(self, repo_path: str | Path, file_path: str | Path, line: int, radius: int = 20) -> CodeWindow:
        repo = Path(repo_path).resolve()
        target = (repo / file_path).resolve()
        target.relative_to(repo)

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        line_start = max(1, line - radius)
        line_end = min(len(lines), line + radius)
        content = "\n".join(lines[line_start - 1 : line_end])

        return CodeWindow(
            file_path=Path(file_path).as_posix(),
            line_start=line_start,
            line_end=line_end,
            content=content,
        )
