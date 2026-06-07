from __future__ import annotations

from abc import ABC, abstractmethod

from codeatlas.ingestion.models import SourceFile
from codeatlas.symbols.models import Symbol


class BaseParser(ABC):
    language: str

    @abstractmethod
    def parse(self, source_file: SourceFile) -> list[Symbol]:
        raise NotImplementedError

