from __future__ import annotations

from codeatlas.chunking.models import CodeChunk
from codeatlas.symbols.models import Symbol


class SymbolProfiler:
    def describe(self, symbol: Symbol, chunk: CodeChunk) -> str:
        first_line = chunk.content.strip().splitlines()[0] if chunk.content.strip() else symbol.name
        return (
            f"{symbol.kind} {symbol.qualified_name} in {symbol.file_path} "
            f"lines {symbol.start_line}-{symbol.end_line}. Signature: {first_line}"
        )

