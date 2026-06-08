from __future__ import annotations

from functools import cache

from tree_sitter import Language, Parser


class TreeSitterManager:
    @cache
    def parser_for(self, language: str) -> Parser:
        if language != "java":
            raise ValueError(f"Unsupported Tree-sitter language: {language}")

        import tree_sitter_java

        return Parser(Language(tree_sitter_java.language()))

