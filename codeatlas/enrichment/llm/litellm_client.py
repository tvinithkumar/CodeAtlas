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
            **self._completion_kwargs(prompt),
        )
        return str(response.choices[0].message.content or "")

    def _completion_kwargs(self, prompt: str) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.base_url:
            kwargs["api_base"] = self.settings.base_url
        if self.settings.api_key:
            kwargs["api_key"] = self.settings.api_key
        elif self.settings.provider == "lmstudio":
            kwargs["api_key"] = "lm-studio"
        return kwargs

    def _model_name(self) -> str:
        if self.settings.provider == "ollama":
            return f"ollama/{self.settings.model}"
        if self.settings.provider == "lmstudio":
            return f"openai/{self.settings.model}"
        return self.settings.model
