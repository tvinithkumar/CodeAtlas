from __future__ import annotations

from codeatlas.ingestion.models import SourceFile
from codeatlas.parsers.python_parser import PythonParser
from codeatlas.symbols.models import Symbol


class SymbolExtractor:
    def __init__(self) -> None:
        self.parsers = {"python": PythonParser()}

    def extract(self, source_file: SourceFile) -> list[Symbol]:
        parser = self.parsers.get(source_file.language)
        if parser is None:
            return []
        return parser.parse(source_file)

