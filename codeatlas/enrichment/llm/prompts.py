from __future__ import annotations

from codeatlas.chunking.models import CodeChunk
from codeatlas.symbols.models import Symbol


def describe_symbol_prompt(symbol: Symbol, chunk: CodeChunk) -> str:
    return f"""You are enriching a code search index.
Return only valid JSON with this schema:
{{
  "description": "one concise paragraph",
  "tags": ["search", "tags"],
  "inputs": ["input names or data dependencies"],
  "outputs": ["return values or side effects"],
  "failure_modes": ["likely failure modes"]
}}

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
    return f"""Return only valid JSON with keys description, tags, inputs, outputs, failure_modes.

Task: summarize_function
Function: {symbol.qualified_name}
File: {symbol.file_path}

Code:
```{symbol.language}
{chunk.content}
```
"""


def generate_search_tags_prompt(symbol: Symbol, chunk: CodeChunk) -> str:
    return f"""Return only valid JSON with keys description, tags, inputs, outputs, failure_modes.
Focus on search tags developers might use to find this code.

Task: generate_search_tags
Symbol: {symbol.qualified_name}
File: {symbol.file_path}

Code:
```{symbol.language}
{chunk.content}
```
"""

