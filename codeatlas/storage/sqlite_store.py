from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from codeatlas.chunking.models import CodeChunk
from codeatlas.enrichment.llm.models import LLMEnrichment
from codeatlas.symbols.models import Relationship, Symbol


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                content TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbols (
                id TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                language TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                parent TEXT,
                UNIQUE (id),
                PRIMARY KEY (qualified_name, file_path)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                unit_type TEXT NOT NULL,
                symbol_qualified_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                profile TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                unit_type,
                symbol_qualified_name,
                file_path,
                content,
                profile
            );

            CREATE TABLE IF NOT EXISTS symbol_edges (
                src_symbol_id TEXT NOT NULL,
                dst_symbol_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (src_symbol_id, dst_symbol_id, edge_type, file_path)
            );

            CREATE TABLE IF NOT EXISTS llm_enrichments (
                chunk_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                tags TEXT NOT NULL,
                inputs TEXT NOT NULL,
                outputs TEXT NOT NULL,
                failure_modes TEXT NOT NULL
            );
            """
        )
        self._ensure_column("symbols", "id", "TEXT")
        self._ensure_column("chunks", "unit_type", "TEXT NOT NULL DEFAULT 'function_chunk'")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def upsert_file(self, path: str, language: str, content: str) -> None:
        self.connection.execute(
            """
            INSERT INTO files(path, language, content)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET language = excluded.language, content = excluded.content
            """,
            (path, language, content),
        )

    def upsert_symbol(self, symbol: Symbol) -> None:
        self.connection.execute(
            """
            INSERT INTO symbols(
                id, qualified_name, file_path, name, kind, language, start_line, end_line, parent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(qualified_name, file_path) DO UPDATE SET
                id = excluded.id,
                name = excluded.name,
                kind = excluded.kind,
                language = excluded.language,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                parent = excluded.parent
            """,
            (
                symbol.id,
                symbol.qualified_name,
                symbol.file_path,
                symbol.name,
                symbol.kind,
                symbol.language,
                symbol.start_line,
                symbol.end_line,
                symbol.parent,
            ),
        )

    def upsert_llm_enrichment(self, chunk_id: str, enrichment: LLMEnrichment) -> None:
        self.connection.execute(
            """
            INSERT INTO llm_enrichments(chunk_id, description, tags, inputs, outputs, failure_modes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                description = excluded.description,
                tags = excluded.tags,
                inputs = excluded.inputs,
                outputs = excluded.outputs,
                failure_modes = excluded.failure_modes
            """,
            (
                chunk_id,
                enrichment.description,
                "\n".join(enrichment.tags),
                "\n".join(enrichment.inputs),
                "\n".join(enrichment.outputs),
                "\n".join(enrichment.failure_modes),
            ),
        )

    def upsert_chunk(self, chunk: CodeChunk, profile: str) -> None:
        self.connection.execute(
            """
            INSERT INTO chunks(
                id, unit_type, symbol_qualified_name, file_path, language, start_line, end_line, content, profile
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                unit_type = excluded.unit_type,
                symbol_qualified_name = excluded.symbol_qualified_name,
                file_path = excluded.file_path,
                language = excluded.language,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                content = excluded.content,
                profile = excluded.profile
            """,
            (
                chunk.id,
                chunk.unit_type,
                chunk.symbol_qualified_name,
                chunk.file_path,
                chunk.language,
                chunk.start_line,
                chunk.end_line,
                chunk.content,
                profile,
            ),
        )
        self.connection.execute("DELETE FROM chunks_fts WHERE id = ?", (chunk.id,))
        self.connection.execute(
            """
            INSERT INTO chunks_fts(id, unit_type, symbol_qualified_name, file_path, content, profile)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk.id, chunk.unit_type, chunk.symbol_qualified_name, chunk.file_path, chunk.content, profile),
        )

    def replace_edges(self, file_path: str, edges: list[Relationship]) -> None:
        self.connection.execute("DELETE FROM symbol_edges WHERE file_path = ?", (file_path,))
        for edge in edges:
            self.connection.execute(
                """
                INSERT INTO symbol_edges(src_symbol_id, dst_symbol_id, edge_type, file_path, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(src_symbol_id, dst_symbol_id, edge_type, file_path) DO UPDATE SET
                    confidence = excluded.confidence
                """,
                (edge.source, edge.target, edge.kind, edge.file_path, edge.confidence),
            )

    def commit(self) -> None:
        self.connection.commit()

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        fts_query = self._fts_query(query)
        if not fts_query:
            return []
        rows = self.connection.execute(
            """
            SELECT c.id, c.file_path, c.symbol_qualified_name, c.content, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def _fts_query(self, query: str) -> str:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
        return " OR ".join(tokens)

    def find_symbols(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT id, qualified_name, file_path, name, kind, language, start_line, end_line, parent
            FROM symbols
            WHERE name = ?
               OR qualified_name = ?
               OR name LIKE ?
               OR qualified_name LIKE ?
            ORDER BY
                CASE WHEN name = ? OR qualified_name = ? THEN 0 ELSE 1 END,
                qualified_name
            LIMIT ?
            """,
            (query, query, f"%{query}%", f"%{query}%", query, query, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_symbol_edges(self, symbol_id: str, direction: str = "both") -> list[dict[str, object]]:
        predicates = {
            "out": "src_symbol_id = ?",
            "in": "dst_symbol_id = ?",
            "both": "src_symbol_id = ? OR dst_symbol_id = ?",
        }
        predicate = predicates[direction]
        params: tuple[str, ...] = (symbol_id, symbol_id) if direction == "both" else (symbol_id,)
        rows = self.connection.execute(
            f"""
            SELECT src_symbol_id, dst_symbol_id, edge_type, file_path, confidence
            FROM symbol_edges
            WHERE {predicate}
            ORDER BY confidence DESC, edge_type
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def edges_by_type(self, edge_type: str, limit: int = 20) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT src_symbol_id, dst_symbol_id, edge_type, file_path, confidence
            FROM symbol_edges
            WHERE edge_type = ?
            ORDER BY confidence DESC
            LIMIT ?
            """,
            (edge_type, limit),
        ).fetchall()
        return [dict(row) for row in rows]
