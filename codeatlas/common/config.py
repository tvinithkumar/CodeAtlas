from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.1
    max_tokens: int = 512
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the local MVP indexer."""

    sqlite_path: Path = Path(".codeatlas/codeatlas.db")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "codeatlas_chunks"
    embedding_dimension: int = 384
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm: LLMSettings = LLMSettings()
    include_globs: tuple[str, ...] = ("*.py",)
    exclude_dirs: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install codeatlas[llm] to load YAML config files.") from exc

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        llm_data = data.get("llm", {})
        return cls(
            sqlite_path=Path(data.get("sqlite_path", cls.sqlite_path)),
            qdrant_url=data.get("qdrant_url", cls.qdrant_url),
            qdrant_collection=data.get("qdrant_collection", cls.qdrant_collection),
            embedding_dimension=data.get("embedding_dimension", cls.embedding_dimension),
            embedding_model=data.get("embedding_model", cls.embedding_model),
            llm=LLMSettings(
                enabled=llm_data.get("enabled", LLMSettings.enabled),
                provider=llm_data.get("provider", LLMSettings.provider),
                model=llm_data.get("model", LLMSettings.model),
                temperature=llm_data.get("temperature", LLMSettings.temperature),
                max_tokens=llm_data.get("max_tokens", LLMSettings.max_tokens),
                base_url=llm_data.get("base_url", LLMSettings.base_url),
                api_key=llm_data.get("api_key", LLMSettings.api_key),
            ),
        )
