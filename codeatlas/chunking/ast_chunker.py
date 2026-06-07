from __future__ import annotations

import uuid

from codeatlas.chunking.models import CodeChunk
from codeatlas.ingestion.models import SourceFile
from codeatlas.symbols.models import Symbol


class ASTChunker:
    def chunk(self, source_file: SourceFile, symbols: list[Symbol]) -> list[CodeChunk]:
        lines = source_file.content.splitlines()
        chunks: list[CodeChunk] = []
        for symbol in symbols:
            content = "\n".join(lines[symbol.start_line - 1 : symbol.end_line])
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file.relative_path}:{symbol.qualified_name}"))
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    unit_type=self._unit_type(symbol),
                    symbol_qualified_name=symbol.qualified_name,
                    file_path=source_file.relative_path,
                    language=source_file.language,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    content=content,
                )
            )
        return chunks

    def _unit_type(self, symbol: Symbol) -> str:
        if symbol.kind == "class":
            return "class_chunk"
        if symbol.kind in {"function", "async_function"}:
            return "function_chunk"
        return "symbol_profile"
