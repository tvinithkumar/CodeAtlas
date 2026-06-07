from __future__ import annotations

import json
from urllib import request

from codeatlas.common.config import LLMSettings


class LMStudioClient:
    """OpenAI-compatible client for LM Studio's local server."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.base_url = (settings.base_url or "http://localhost:1234/v1").rstrip("/")
        self.api_key = settings.api_key or "lm-studio"

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content", ""))

