from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings


class FileDiscovery:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def discover(self, repo_path: Path) -> list[Path]:
        repo_path = repo_path.resolve()
        files: list[Path] = []
        for pattern in self.settings.include_globs:
            for path in repo_path.rglob(pattern):
                if path.is_file() and not self._is_excluded(path, repo_path):
                    files.append(path)
        return sorted(files)

    def _is_excluded(self, path: Path, repo_path: Path) -> bool:
        relative_parts = path.relative_to(repo_path).parts
        return any(part in self.settings.exclude_dirs for part in relative_parts)

