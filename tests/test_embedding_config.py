from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.embedding.factory import build_embedding_provider
from codeatlas.embedding.hash_provider import HashEmbeddingProvider


def test_embedding_config_loads_sentence_transformers_provider(tmp_path: Path) -> None:
    config = tmp_path / "codeatlas.yml"
    config.write_text(
        """
embeddings:
  provider: sentence_transformers
  model: jinaai/jina-embeddings-v2-base-code
  dimensions: 768
  batch_size: 32
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config)

    assert settings.embeddings.provider == "sentence_transformers"
    assert settings.embeddings.model == "jinaai/jina-embeddings-v2-base-code"
    assert settings.embeddings.dimensions == 768
    assert settings.embeddings.batch_size == 32


def test_hash_embedding_provider_factory_is_offline_safe() -> None:
    settings = Settings.from_dict(
        {
            "embeddings": {
                "provider": "hash",
                "dimensions": 8,
            }
        }
    )

    provider = build_embedding_provider(settings.embeddings)
    vector = provider.embed("retry backoff")

    assert isinstance(provider, HashEmbeddingProvider)
    assert len(vector) == 8

