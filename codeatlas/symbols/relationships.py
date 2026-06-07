from __future__ import annotations

import ast
from collections.abc import Iterable

from codeatlas.ingestion.models import SourceFile
from codeatlas.symbols.models import Relationship, Symbol


class PythonRelationshipExtractor:
    def extract(self, source_file: SourceFile, symbols: list[Symbol]) -> list[Relationship]:
        tree = ast.parse(source_file.content, filename=source_file.relative_path)
        by_range = sorted(symbols, key=lambda item: (item.start_line, item.end_line))
        by_name = {symbol.name: symbol.qualified_name for symbol in symbols}
        by_qualified = {symbol.qualified_name: symbol.qualified_name for symbol in symbols}
        edges: list[Relationship] = []

        for node in ast.walk(tree):
            owner = self._owner(node, by_range)
            if owner is None:
                continue

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for imported_name in self._import_names(node):
                    edges.append(Relationship(owner.qualified_name, imported_name, "imports", source_file.relative_path, 0.8))

            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = self._name_for(base)
                    if base_name:
                        edges.append(Relationship(owner.qualified_name, base_name, "inherits", source_file.relative_path, 0.9))

            if isinstance(node, ast.Call):
                call_name = self._name_for(node.func)
                if not call_name:
                    continue
                edge_type = self._call_edge_type(call_name)
                target = by_name.get(call_name) or by_qualified.get(call_name) or call_name
                edges.append(Relationship(owner.qualified_name, target, edge_type, source_file.relative_path, 0.75))

            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and self._looks_like_config(node.id):
                edges.append(Relationship(owner.qualified_name, node.id, "reads_config", source_file.relative_path, 0.55))

        return self._dedupe(edges)

    def _owner(self, node: ast.AST, symbols: list[Symbol]) -> Symbol | None:
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return None
        matches = [symbol for symbol in symbols if symbol.start_line <= lineno <= symbol.end_line]
        if not matches:
            return None
        return max(matches, key=lambda item: item.start_line)

    def _import_names(self, node: ast.Import | ast.ImportFrom) -> Iterable[str]:
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        module = node.module or ""
        return [f"{module}.{alias.name}" if module else alias.name for alias in node.names]

    def _name_for(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._name_for(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def _call_edge_type(self, call_name: str) -> str:
        lowered = call_name.lower()
        if "log" in lowered or lowered in {"debug", "info", "warning", "error", "exception"}:
            return "logs"
        if "metric" in lowered or "counter" in lowered or "histogram" in lowered or "gauge" in lowered:
            return "emits_metric"
        if "query" in lowered or "execute" in lowered or "fetch" in lowered:
            return "queries_table"
        return "calls"

    def _looks_like_config(self, name: str) -> bool:
        lowered = name.lower()
        return any(part in lowered for part in ("config", "timeout", "retry", "backoff", "env"))

    def _dedupe(self, edges: list[Relationship]) -> list[Relationship]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[Relationship] = []
        for edge in edges:
            key = (edge.source, edge.target, edge.kind, edge.file_path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(edge)
        return deduped


class RelationshipExtractor:
    def __init__(self) -> None:
        self.extractors = {"python": PythonRelationshipExtractor()}

    def extract(self, source_file: SourceFile, symbols: list[Symbol]) -> list[Relationship]:
        extractor = self.extractors.get(source_file.language)
        if extractor is None:
            return []
        return extractor.extract(source_file, symbols)
