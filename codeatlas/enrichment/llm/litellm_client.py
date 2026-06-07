from __future__ import annotations

from codeatlas.common.config import LLMSettings


class LiteLLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def complete(self, prompt: str) -> str:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError("Install codeatlas[llm] to enable LiteLLM enrichment.") from exc

        response = completion(
            model=self._model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.temperature,
        )
        return str(response.choices[0].message.content or "")

    def _model_name(self) -> str:
        if self.settings.provider == "ollama":
            return f"ollama/{self.settings.model}"
        return self.settings.model

