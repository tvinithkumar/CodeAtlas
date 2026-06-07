from __future__ import annotations


class FastEmbedProvider:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return list(next(self.model.embed([text])))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self.model.embed(texts)]

