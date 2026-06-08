from __future__ import annotations

import ast
from collections.abc import Iterable

from codeatlas.ingestion.models import SourceFile
from codeatlas.parsers.tree_sitter_manager import TreeSitterManager
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
        seen: set[tuple[str, str, str, str, int | None]] = set()
        deduped: list[Relationship] = []
        for edge in edges:
            key = (edge.source, edge.target, edge.kind, edge.file_path, edge.line_start)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(edge)
        return deduped


class JavaRelationshipExtractor:
    def __init__(self, manager: TreeSitterManager | None = None) -> None:
        self.manager = manager or TreeSitterManager()

    def extract(self, source_file: SourceFile, symbols: list[Symbol]) -> list[Relationship]:
        parser = self.manager.parser_for("java")
        tree = parser.parse(source_file.content.encode("utf-8"))
        by_range = sorted(symbols, key=lambda item: (item.start_line, item.end_line))
        by_name = {symbol.name: symbol.qualified_name for symbol in symbols}
        edges: list[Relationship] = []
        import_owner = self._import_owner(symbols)

        for node in self._walk(tree.root_node):
            owner = self._owner(node, by_range)

            if node.type == "import_declaration" and import_owner is not None:
                imported_name = self._import_name(node, source_file.content)
                if imported_name:
                    edges.append(self._edge(import_owner, imported_name, "imports", source_file, node, 0.8))
                continue

            if node.type in {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}:
                class_symbol = self._owner(node, by_range)
                if class_symbol is not None:
                    for parent_name in self._inheritance_names(node, source_file.content):
                        edges.append(self._edge(class_symbol, parent_name, "inherits", source_file, node, 0.9))
                continue

            if owner is None:
                continue

            if node.type == "method_invocation":
                call_name = self._method_invocation_name(node, source_file.content)
                if call_name:
                    target = by_name.get(call_name) or call_name
                    edges.append(self._edge(owner, target, "calls", source_file, node, 0.75))
                continue

            if node.type == "identifier":
                name = self._text(node, source_file.content)
                if self._should_record_identifier_use(name, owner, node):
                    target = by_name.get(name) or name
                    edges.append(self._edge(owner, target, "uses", source_file, node, 0.45))

        return self._dedupe(edges)

    def _edge(
        self,
        source: Symbol,
        target: str,
        kind: str,
        source_file: SourceFile,
        node,
        confidence: float,
    ) -> Relationship:
        return Relationship(
            source.qualified_name,
            target,
            kind,
            source_file.relative_path,
            confidence,
            node.start_point[0] + 1,
            node.end_point[0] + 1,
        )

    def _owner(self, node, symbols: list[Symbol]) -> Symbol | None:
        line = node.start_point[0] + 1
        matches = [
            symbol
            for symbol in symbols
            if symbol.start_line <= line <= symbol.end_line and symbol.kind in {"class", "method", "constructor"}
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.start_line, -item.end_line))

    def _import_owner(self, symbols: list[Symbol]) -> Symbol | None:
        classes = [symbol for symbol in symbols if symbol.kind == "class"]
        if classes:
            return min(classes, key=lambda item: item.start_line)
        return symbols[0] if symbols else None

    def _import_name(self, node, content: str) -> str | None:
        for child in node.children:
            if child.type in {"scoped_identifier", "identifier", "asterisk"}:
                return self._text(child, content)
        return None

    def _inheritance_names(self, node, content: str) -> list[str]:
        names: list[str] = []
        for child in node.children:
            if child.type in {"superclass", "super_interfaces"}:
                names.extend(
                    self._text(descendant, content)
                    for descendant in self._walk(child)
                    if descendant.type in {"type_identifier", "scoped_type_identifier", "generic_type"}
                )
        return names

    def _method_invocation_name(self, node, content: str) -> str | None:
        name = node.child_by_field_name("name")
        if name is not None:
            return self._text(name, content)
        identifiers = [child for child in node.children if child.type == "identifier"]
        if identifiers:
            return self._text(identifiers[-1], content)
        return None

    def _should_record_identifier_use(self, name: str, owner: Symbol, node) -> bool:
        if name == owner.name:
            return False
        if node.parent is not None and node.parent.type in {
            "class_declaration",
            "constructor_declaration",
            "formal_parameter",
            "import_declaration",
            "method_declaration",
            "package_declaration",
            "variable_declarator",
        }:
            return False
        return bool(name) and not name[0].islower()

    def _text(self, node, content: str) -> str:
        return content.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8")

    def _walk(self, node):
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _dedupe(self, edges: list[Relationship]) -> list[Relationship]:
        seen: set[tuple[str, str, str, str, int | None]] = set()
        deduped: list[Relationship] = []
        for edge in edges:
            key = (edge.source, edge.target, edge.kind, edge.file_path, edge.line_start)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(edge)
        return deduped


class RelationshipExtractor:
    def __init__(self) -> None:
        self.extractors = {"python": PythonRelationshipExtractor(), "java": JavaRelationshipExtractor()}

    def extract(self, source_file: SourceFile, symbols: list[Symbol]) -> list[Relationship]:
        extractor = self.extractors.get(source_file.language)
        if extractor is None:
            return []
        return extractor.extract(source_file, symbols)
