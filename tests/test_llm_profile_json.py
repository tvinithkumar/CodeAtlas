from __future__ import annotations

from codeatlas.enrichment.llm.enricher import LLMEnricher
from codeatlas.enrichment.llm.json_parser import LLMProfileJSONParser


class TimeoutClient:
    def complete(self, prompt: str) -> str:
        raise TimeoutError("timed out")


def test_profile_parser_accepts_valid_json() -> None:
    profile = LLMProfileJSONParser().parse(
        """
{
  "description": "Configures retry backoff.",
  "responsibilities": ["retry timing"],
  "inputs": ["attempt"],
  "outputs": ["delay"],
  "side_effects": ["none"],
  "failure_modes": ["negative attempt"],
  "search_tags": ["retry", "backoff"]
}
""".strip()
    )

    assert profile.description == "Configures retry backoff."
    assert profile.responsibilities == ["retry timing"]
    assert profile.search_tags == ["retry", "backoff"]


def test_profile_parser_accepts_markdown_wrapped_json() -> None:
    profile = LLMProfileJSONParser().parse(
        """
```json
{
  "description": "Emits latency metrics.",
  "responsibilities": ["record latency"],
  "inputs": ["duration"],
  "outputs": [],
  "side_effects": ["metric emission"],
  "failure_modes": [],
  "search_tags": ["latency", "metrics"]
}
```
""".strip()
    )

    assert profile.description == "Emits latency metrics."
    assert profile.side_effects == ["metric emission"]


def test_profile_parser_returns_empty_profile_for_invalid_json() -> None:
    profile = LLMProfileJSONParser().parse("not json")

    assert profile.is_empty is True


def test_enricher_returns_empty_profile_on_timeout() -> None:
    profile = LLMEnricher(TimeoutClient())._complete("prompt")

    assert profile.is_empty is True

