from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.ingestion.file_discovery import FileDiscovery
from codeatlas.ingestion.models import SourceFile


class RepositoryLoader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.discovery = FileDiscovery(self.settings)

    def load(self, repo_path: Path) -> list[SourceFile]:
        repo_path = repo_path.resolve()
        source_files: list[SourceFile] = []
        for path in self.discovery.discover(repo_path):
            source_files.append(
                SourceFile(
                    path=path,
                    relative_path=path.relative_to(repo_path).as_posix(),
                    language=self._language_for(path),
                    content=path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        return source_files

    def _language_for(self, path: Path) -> str:
        if path.suffix == ".py":
            return "python"
        if path.suffix == ".java":
            return "java"
        return path.suffix.lstrip(".") or "text"
