from __future__ import annotations

from codeatlas.storage.sqlite_store import SQLiteStore


class GraphSearch:
    def __init__(self, sqlite_store: SQLiteStore) -> None:
        self.sqlite_store = sqlite_store

    def explain_symbol(self, symbol_query: str) -> dict[str, object] | None:
        symbols = self.sqlite_store.find_symbols(symbol_query, limit=1)
        if not symbols:
            return None
        symbol = symbols[0]
        edges = self.sqlite_store.get_symbol_edges(str(symbol["id"]), direction="both")
        return {"symbol": symbol, "edges": edges}

    def find_usage(self, symbol_query: str) -> list[dict[str, object]]:
        symbols = self.sqlite_store.find_symbols(symbol_query, limit=10)
        usages: list[dict[str, object]] = []
        for symbol in symbols:
            usages.extend(self.sqlite_store.get_symbol_edges(str(symbol["id"]), direction="in"))
        return usages

    def impact_analysis(self, symbol_query: str, depth: int = 2) -> dict[str, object] | None:
        symbols = self.sqlite_store.find_symbols(symbol_query, limit=1)
        if not symbols:
            return None
        root = symbols[0]
        visited = {str(root["id"])}
        frontier = [str(root["id"])]
        edges: list[dict[str, object]] = []

        for _ in range(depth):
            next_frontier: list[str] = []
            for symbol_id in frontier:
                incoming = self.sqlite_store.get_symbol_edges(symbol_id, direction="in")
                edges.extend(incoming)
                for edge in incoming:
                    src = str(edge["src_symbol_id"])
                    if src not in visited:
                        visited.add(src)
                        next_frontier.append(src)
            frontier = next_frontier
            if not frontier:
                break

        return {"symbol": root, "impacted_symbol_ids": sorted(visited - {str(root["id"])}), "edges": edges}

    def architecture_search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        lowered = query.lower()
        if "metric" in lowered or "latency" in lowered:
            return self.sqlite_store.edges_by_type("emits_metric", limit=limit)
        if "log" in lowered:
            return self.sqlite_store.edges_by_type("logs", limit=limit)
        if "table" in lowered or "query" in lowered or "database" in lowered:
            return self.sqlite_store.edges_by_type("queries_table", limit=limit)
        if "config" in lowered or "timeout" in lowered or "retry" in lowered:
            return self.sqlite_store.edges_by_type("reads_config", limit=limit)
        return []

