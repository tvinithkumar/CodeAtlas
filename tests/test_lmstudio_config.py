from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from codeatlas.common.config import Settings
from codeatlas.enrichment.llm.lmstudio_client import LMStudioClient
from codeatlas.enrichment.llm.litellm_client import LiteLLMClient


def test_lmstudio_config_loads_base_url(tmp_path: Path) -> None:
    config = tmp_path / "codeatlas.yml"
    config.write_text(
        """
llm:
  enabled: true
  provider: lmstudio
  model: qwen3-coder-30b-a3b-instruct
  temperature: 0.1
  max_tokens: 512
  base_url: http://localhost:1234/v1
  api_key: lm-studio
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config)

    assert settings.llm.enabled is True
    assert settings.llm.provider == "lmstudio"
    assert settings.llm.model == "qwen3-coder-30b-a3b-instruct"
    assert settings.llm.max_tokens == 512
    assert settings.llm.base_url == "http://localhost:1234/v1"
    assert settings.llm.api_key == "lm-studio"


def test_lmstudio_litellm_kwargs_use_openai_compatible_endpoint() -> None:
    settings = Settings.from_dict(
        {
            "llm": {
                "enabled": True,
                "provider": "lmstudio",
                "model": "qwen3-coder-30b-a3b-instruct",
                "temperature": 0.1,
                "max_tokens": 512,
                "base_url": "http://localhost:1234/v1",
            }
        }
    )
    client = LiteLLMClient(settings.llm)

    kwargs = client._completion_kwargs("hello")

    assert kwargs["model"] == "openai/qwen3-coder-30b-a3b-instruct"
    assert kwargs["api_base"] == "http://localhost:1234/v1"
    assert kwargs["api_key"] == "lm-studio"
    assert kwargs["max_tokens"] == 512
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_lmstudio_client_posts_openai_compatible_payload() -> None:
    settings = Settings.from_dict(
        {
            "llm": {
                "enabled": True,
                "provider": "lmstudio",
                "model": "qwen3-coder-30b-a3b-instruct",
                "temperature": 0.1,
                "max_tokens": 256,
                "base_url": "http://localhost:1234/v1",
            }
        }
    )
    response = MagicMock()
    response.__enter__.return_value.read.return_value = (
        b'{"choices":[{"message":{"content":"description text"}}]}'
    )

    with patch("codeatlas.enrichment.llm.lmstudio_client.request.urlopen", return_value=response) as urlopen:
        content = LMStudioClient(settings.llm).complete("describe this")

    req = urlopen.call_args.args[0]
    assert content == "description text"
    assert req.full_url == "http://localhost:1234/v1/chat/completions"
    assert b'"model": "qwen3-coder-30b-a3b-instruct"' in req.data
    assert b'"max_tokens": 256' in req.data
