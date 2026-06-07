from __future__ import annotations

import ast

from codeatlas.ingestion.models import SourceFile
from codeatlas.parsers.base_parser import BaseParser
from codeatlas.symbols.models import Symbol


class PythonParser(BaseParser):
    language = "python"

    def parse(self, source_file: SourceFile) -> list[Symbol]:
        tree = ast.parse(source_file.content, filename=source_file.relative_path)
        symbols: list[Symbol] = []
        self._walk(tree, source_file, symbols, parent=None)
        return symbols

    def _walk(
        self,
        node: ast.AST,
        source_file: SourceFile,
        symbols: list[Symbol],
        parent: str | None,
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qualified = self._qualified(parent, child.name)
                symbols.append(self._symbol(child, source_file, "class", qualified, parent))
                self._walk(child, source_file, symbols, qualified)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = self._qualified(parent, child.name)
                kind = "async_function" if isinstance(child, ast.AsyncFunctionDef) else "function"
                symbols.append(self._symbol(child, source_file, kind, qualified, parent))
                self._walk(child, source_file, symbols, qualified)
            elif parent is None and isinstance(child, (ast.Assign, ast.AnnAssign)):
                for name in self._assigned_names(child):
                    symbols.append(
                        Symbol(
                            name=name,
                            kind="variable",
                            file_path=source_file.relative_path,
                            language=source_file.language,
                            start_line=child.lineno,
                            end_line=getattr(child, "end_lineno", child.lineno),
                            qualified_name=name,
                            parent=None,
                        )
                    )
            else:
                self._walk(child, source_file, symbols, parent)

    def _symbol(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        source_file: SourceFile,
        kind: str,
        qualified_name: str,
        parent: str | None,
    ) -> Symbol:
        return Symbol(
            name=node.name,
            kind=kind,
            file_path=source_file.relative_path,
            language=source_file.language,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            qualified_name=qualified_name,
            parent=parent,
        )

    def _qualified(self, parent: str | None, name: str) -> str:
        return f"{parent}.{name}" if parent else name

    def _assigned_names(self, node: ast.Assign | ast.AnnAssign) -> list[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names: list[str] = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
        return names
