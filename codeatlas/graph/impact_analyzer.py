from __future__ import annotations

from pathlib import Path
from typing import Any

from codeatlas.retrieval.code_window import CodeWindowFetcher
from codeatlas.storage.sqlite_store import SQLiteStore


class ImpactAnalyzer:
    def __init__(self, sqlite_store: SQLiteStore, window_fetcher: CodeWindowFetcher | None = None) -> None:
        self.sqlite_store = sqlite_store
        self.window_fetcher = window_fetcher or CodeWindowFetcher()

    def analyze(
        self,
        symbol_query: str,
        repo_path: str | Path | None = None,
        depth: int = 2,
        limit: int = 10,
        window_radius: int = 12,
    ) -> dict[str, Any] | None:
        symbols = self.sqlite_store.find_symbols(symbol_query, limit=1)
        if not symbols:
            return None

        root = symbols[0]
        root_id = str(root["id"])
        direct_callers = self._incoming_edges(root_id, edge_type="calls", limit=limit)
        incoming_related = self._incoming_edges(root_id, limit=limit)
        outgoing_related = self._outgoing_edges(root_id, limit=limit)
        same_file_symbols = self._same_file_symbols(root, limit=limit)
        affected_files = self._affected_files(root, direct_callers, incoming_related, outgoing_related)
        windows = self._code_windows(root, direct_callers, repo_path, window_radius, limit)

        return {
            "symbol": root,
            "direct_callers": direct_callers,
            "related_methods": self._related_methods(root, incoming_related, outgoing_related, same_file_symbols, limit),
            "files_affected": affected_files,
            "risk_notes": self._risk_notes(root, direct_callers, incoming_related, outgoing_related, affected_files),
            "top_code_windows": windows,
            "impact_graph": self._walk_incoming(root_id, depth=depth, limit=limit),
        }

    def _incoming_edges(self, symbol_id: str, edge_type: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        edge_filter = "AND e.edge_type = ?" if edge_type else ""
        params: tuple[Any, ...] = (symbol_id, edge_type, limit) if edge_type else (symbol_id, limit)
        rows = self.sqlite_store.connection.execute(
            f"""
            SELECT
                e.src_symbol_id,
                src.qualified_name AS src_symbol,
                src.kind AS src_kind,
                src.file_path AS src_file_path,
                src.start_line AS src_start_line,
                src.end_line AS src_end_line,
                e.dst_symbol_id,
                dst.qualified_name AS dst_symbol,
                e.edge_type,
                e.file_path,
                e.line_start,
                e.line_end,
                e.confidence
            FROM symbol_edges e
            LEFT JOIN symbols src ON src.id = e.src_symbol_id
            LEFT JOIN symbols dst ON dst.id = e.dst_symbol_id
            WHERE e.dst_symbol_id = ?
            {edge_filter}
            ORDER BY e.confidence DESC, e.line_start
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def _outgoing_edges(self, symbol_id: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.sqlite_store.connection.execute(
            """
            SELECT
                e.src_symbol_id,
                src.qualified_name AS src_symbol,
                e.dst_symbol_id,
                dst.qualified_name AS dst_symbol,
                e.edge_type,
                e.file_path,
                e.line_start,
                e.line_end,
                e.confidence
            FROM symbol_edges e
            LEFT JOIN symbols src ON src.id = e.src_symbol_id
            LEFT JOIN symbols dst ON dst.id = e.dst_symbol_id
            WHERE e.src_symbol_id = ?
            ORDER BY e.confidence DESC, e.line_start
            LIMIT ?
            """,
            (symbol_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def _same_file_symbols(self, root: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
        rows = self.sqlite_store.connection.execute(
            """
            SELECT id, qualified_name, file_path, name, kind, language, start_line, end_line, parent
            FROM symbols
            WHERE file_path = ?
              AND id != ?
              AND kind IN ('function', 'async_function', 'method', 'constructor', 'class')
            ORDER BY
                ABS(start_line - ?) ASC,
                start_line ASC
            LIMIT ?
            """,
            (root["file_path"], root["id"], root["start_line"], limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def _affected_files(
        self,
        root: dict[str, Any],
        direct_callers: list[dict[str, Any]],
        incoming_related: list[dict[str, Any]],
        outgoing_related: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}

        def add(file_path: str | None, reason: str, weight: float) -> None:
            if not file_path:
                return
            item = scores.setdefault(file_path, {"file_path": file_path, "score": 0.0, "reasons": []})
            item["score"] += weight
            if reason not in item["reasons"]:
                item["reasons"].append(reason)

        add(str(root["file_path"]), "target symbol is defined here", 2.0)
        for edge in direct_callers:
            add(edge.get("src_file_path") or edge.get("file_path"), "direct caller", 1.5)
        for edge in incoming_related:
            add(edge.get("src_file_path") or edge.get("file_path"), f"incoming {edge['edge_type']} edge", 1.0)
        for edge in outgoing_related:
            add(edge.get("file_path"), f"outgoing {edge['edge_type']} edge", 0.5)

        return sorted(scores.values(), key=lambda item: (-item["score"], item["file_path"]))

    def _related_methods(
        self,
        root: dict[str, Any],
        incoming_related: list[dict[str, Any]],
        outgoing_related: list[dict[str, Any]],
        same_file_symbols: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        related: dict[str, dict[str, Any]] = {}

        def add(symbol_id: str | None, symbol: str | None, relation: str, file_path: str | None = None) -> None:
            if not symbol_id and not symbol:
                return
            key = symbol_id or str(symbol)
            if key == root["id"]:
                return
            item = related.setdefault(key, {"symbol_id": symbol_id, "symbol": symbol or symbol_id, "relations": []})
            if file_path:
                item["file_path"] = file_path
            if relation not in item["relations"]:
                item["relations"].append(relation)

        for edge in incoming_related:
            add(edge.get("src_symbol_id"), edge.get("src_symbol"), f"incoming {edge['edge_type']}", edge.get("src_file_path"))
        for edge in outgoing_related:
            add(edge.get("dst_symbol_id"), edge.get("dst_symbol") or edge.get("dst_symbol_id"), f"outgoing {edge['edge_type']}", edge.get("file_path"))
        for symbol in same_file_symbols:
            add(str(symbol["id"]), str(symbol["qualified_name"]), "same file", str(symbol["file_path"]))

        return list(related.values())[:limit]

    def _risk_notes(
        self,
        root: dict[str, Any],
        direct_callers: list[dict[str, Any]],
        incoming_related: list[dict[str, Any]],
        outgoing_related: list[dict[str, Any]],
        affected_files: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if direct_callers:
            notes.append(f"{len(direct_callers)} direct caller(s) may need regression coverage.")
        if any(edge["edge_type"] == "calls" for edge in outgoing_related):
            notes.append("Target calls other symbols; validate downstream parsing and conversion behavior.")
        if any("test/" in str(edge.get("src_file_path") or edge.get("file_path")) for edge in incoming_related):
            notes.append("Tests already reference this symbol; failing tests can guide localization.")
        if len(affected_files) > 1:
            notes.append(f"Impact crosses {len(affected_files)} files.")
        if str(root.get("kind")) in {"method", "function", "async_function"} and not direct_callers:
            notes.append("No indexed direct callers found; impact may come from reflection, framework entrypoints, or unresolved calls.")
        return notes

    def _code_windows(
        self,
        root: dict[str, Any],
        direct_callers: list[dict[str, Any]],
        repo_path: str | Path | None,
        radius: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        if repo_path is None:
            return []

        candidates: list[tuple[str, int, str]] = [
            (str(root["file_path"]), int(root["start_line"]), "target definition")
        ]
        for edge in direct_callers:
            line = edge.get("line_start") or edge.get("src_start_line")
            file_path = edge.get("file_path") or edge.get("src_file_path")
            if file_path and line:
                candidates.append((str(file_path), int(line), f"direct caller: {edge.get('src_symbol')}"))

        windows: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for file_path, line, reason in candidates:
            key = (file_path, line)
            if key in seen:
                continue
            seen.add(key)
            try:
                window = self.window_fetcher.get_code_window(repo_path, file_path, line, radius)
            except (OSError, ValueError):
                continue
            data = window.__dict__.copy()
            data["reason"] = reason
            windows.append(data)
            if len(windows) >= limit:
                break
        return windows

    def _walk_incoming(self, root_id: str, depth: int, limit: int) -> list[dict[str, Any]]:
        visited = {root_id}
        frontier = [root_id]
        graph: list[dict[str, Any]] = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for symbol_id in frontier:
                for edge in self._incoming_edges(symbol_id, edge_type="calls", limit=limit):
                    graph.append(edge)
                    src = str(edge["src_symbol_id"])
                    if src not in visited:
                        visited.add(src)
                        next_frontier.append(src)
            frontier = next_frontier
            if not frontier:
                break
        return graph[:limit]
