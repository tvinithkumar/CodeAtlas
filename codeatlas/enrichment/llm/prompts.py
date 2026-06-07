from __future__ import annotations

from codeatlas.chunking.models import CodeChunk
from codeatlas.enrichment.llm.profile_schema import PROFILE_SCHEMA_TEXT
from codeatlas.symbols.models import Symbol


def describe_symbol_prompt(symbol: Symbol, chunk: CodeChunk) -> str:
    return f"""You are enriching a code search index.
Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{PROFILE_SCHEMA_TEXT}

Task: describe_symbol
Symbol: {symbol.qualified_name}
Kind: {symbol.kind}
File: {symbol.file_path}
Lines: {symbol.start_line}-{symbol.end_line}

Code:
```{symbol.language}
{chunk.content}
```
"""


def summarize_function_prompt(symbol: Symbol, chunk: CodeChunk) -> str:
    return f"""Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{PROFILE_SCHEMA_TEXT}

Task: summarize_function
Function: {symbol.qualified_name}
File: {symbol.file_path}

Code:
```{symbol.language}
{chunk.content}
```
"""


def generate_search_tags_prompt(symbol: Symbol, chunk: CodeChunk) -> str:
    return f"""Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{PROFILE_SCHEMA_TEXT}
Focus on search tags developers might use to find this code.

Task: generate_search_tags
Symbol: {symbol.qualified_name}
File: {symbol.file_path}

Code:
```{symbol.language}
{chunk.content}
```
"""
