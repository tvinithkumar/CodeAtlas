from __future__ import annotations

import argparse
from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.mcp.limits import MAX_LINES_PER_RESULT, MAX_RESULTS, MAX_TOTAL_CHARS
from codeatlas.mcp.tools import CodeAtlasMCPTools


def build_server(tools: CodeAtlasMCPTools):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install the optional MCP dependency to run the server: pip install 'codeatlas[mcp]'") from exc

    server = FastMCP("CodeAtlas")

    @server.tool()
    def search_code(query: str, top_k: int = MAX_RESULTS) -> dict:
        return tools.search_code(query, top_k=top_k)

    @server.tool()
    def get_code_window(file: str, line: int, radius: int = 20) -> dict:
        return tools.get_code_window(file=file, line=line, radius=radius)

    @server.tool()
    def explain_symbol(symbol: str) -> dict:
        return tools.explain_symbol(symbol)

    @server.tool()
    def find_usages(symbol: str) -> dict:
        return tools.find_usages(symbol)

    @server.tool()
    def related_symbols(symbol: str) -> dict:
        return tools.related_symbols(symbol)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CodeAtlas MCP server.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sqlite-path", type=Path, default=Settings().sqlite_path)
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS)
    parser.add_argument("--max-lines-per-result", type=int, default=MAX_LINES_PER_RESULT)
    parser.add_argument("--max-total-chars", type=int, default=MAX_TOTAL_CHARS)
    args = parser.parse_args()

    tools = CodeAtlasMCPTools(
        repo_path=args.repo,
        sqlite_path=args.sqlite_path,
        max_results=args.max_results,
        max_lines_per_result=args.max_lines_per_result,
        max_total_chars=args.max_total_chars,
    )
    build_server(tools).run()


if __name__ == "__main__":
    main()

