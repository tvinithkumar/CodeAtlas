from __future__ import annotations

from tree_sitter import Node

from codeatlas.ingestion.models import SourceFile
from codeatlas.parsers.base_parser import BaseParser
from codeatlas.parsers.tree_sitter_manager import TreeSitterManager
from codeatlas.symbols.models import Symbol


class JavaParser(BaseParser):
    language = "java"

    def __init__(self, manager: TreeSitterManager | None = None) -> None:
        self.manager = manager or TreeSitterManager()

    def parse(self, source_file: SourceFile) -> list[Symbol]:
        parser = self.manager.parser_for("java")
        tree = parser.parse(source_file.content.encode("utf-8"))
        package_name = self._package_name(tree.root_node, source_file.content)
        symbols: list[Symbol] = []
        self._walk(tree.root_node, source_file, symbols, parent=package_name)
        return symbols

    def _walk(
        self,
        node: Node,
        source_file: SourceFile,
        symbols: list[Symbol],
        parent: str | None,
    ) -> None:
        if node.type in {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}:
            name = self._field_text(node, "name", source_file.content)
            if name:
                qualified = self._qualified(parent, name)
                symbols.append(self._symbol(node, source_file, "class", name, qualified, parent))
                body = node.child_by_field_name("body")
                if body is not None:
                    self._walk(body, source_file, symbols, qualified)
                return

        if node.type in {"method_declaration", "constructor_declaration"}:
            name = self._field_text(node, "name", source_file.content)
            if name:
                qualified = self._qualified(parent, name)
                kind = "constructor" if node.type == "constructor_declaration" else "method"
                symbols.append(self._symbol(node, source_file, kind, name, qualified, parent))
                return

        if node.type == "field_declaration":
            for declarator in self._children_of_type(node, "variable_declarator"):
                name = self._field_text(declarator, "name", source_file.content)
                if name:
                    qualified = self._qualified(parent, name)
                    symbols.append(self._symbol(node, source_file, "field", name, qualified, parent))

        for child in node.children:
            self._walk(child, source_file, symbols, parent)

    def _symbol(
        self,
        node: Node,
        source_file: SourceFile,
        kind: str,
        name: str,
        qualified_name: str,
        parent: str | None,
    ) -> Symbol:
        return Symbol(
            name=name,
            kind=kind,
            file_path=source_file.relative_path,
            language=source_file.language,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            qualified_name=qualified_name,
            parent=parent,
        )

    def _package_name(self, root: Node, content: str) -> str | None:
        for child in root.children:
            if child.type == "package_declaration":
                scoped_identifier = self._first_named_descendant(child, {"scoped_identifier", "identifier"})
                if scoped_identifier is not None:
                    return self._text(scoped_identifier, content)
        return None

    def _field_text(self, node: Node, field: str, content: str) -> str | None:
        child = node.child_by_field_name(field)
        return self._text(child, content) if child is not None else None

    def _text(self, node: Node, content: str) -> str:
        return content.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8")

    def _qualified(self, parent: str | None, name: str) -> str:
        return f"{parent}.{name}" if parent else name

    def _children_of_type(self, node: Node, node_type: str) -> list[Node]:
        matches: list[Node] = []
        for child in node.children:
            if child.type == node_type:
                matches.append(child)
            matches.extend(self._children_of_type(child, node_type))
        return matches

    def _first_named_descendant(self, node: Node, node_types: set[str]) -> Node | None:
        if node.type in node_types:
            return node
        for child in node.children:
            result = self._first_named_descendant(child, node_types)
            if result is not None:
                return result
        return None
