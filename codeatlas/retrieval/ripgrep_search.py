from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

from codeatlas.storage.models import RetrievalChunk, SearchHit


class RipgrepRetriever:
    def search(
        self,
        repo_path: str | Path,
        query: str,
        context_lines: int = 10,
        max_results: int = 50,
    ) -> list[RetrievalChunk]:
        if shutil.which("rg") is None:
            return []

        cmd = [
            "rg",
            "-n",
            f"-C{context_lines}",
            "--json",
            query,
            str(repo_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode not in {0, 1}:
            return []
        return self._parse_json_lines(result.stdout, Path(repo_path), max_results=max_results)

    def search_hits(
        self,
        repo_path: str | Path,
        query: str,
        context_lines: int = 10,
        max_results: int = 50,
    ) -> list[SearchHit]:
        chunks = self.search(repo_path, query, context_lines=context_lines, max_results=max_results)
        return [
            SearchHit(
                id=chunk.chunk_id,
                score=1.0,
                file_path=chunk.file_path,
                symbol=chunk.file_path,
                content=chunk.content,
                source=chunk.source,
                retrieval_method=chunk.retrieval_method,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )
            for chunk in chunks
        ]

    def _parse_json_lines(self, output: str, repo_path: Path, max_results: int) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []
        pending_context: list[tuple[int, str]] = []
        active_chunk_index: int | None = None

        for line in output.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            data = event.get("data", {})
            if event_type == "begin":
                pending_context = []
                active_chunk_index = None
                continue
            if event_type == "context":
                context_line = self._line_event(data)
                if context_line is None:
                    continue
                if active_chunk_index is None:
                    pending_context.append(context_line)
                else:
                    chunks[active_chunk_index] = self._append_context(chunks[active_chunk_index], context_line)
                continue
            if event_type == "match":
                match_line = self._line_event(data)
                if match_line is None:
                    continue
                file_path = self._relative_path(data, repo_path)
                lines = [*pending_context, match_line]
                chunk = self._chunk(file_path, lines)
                chunks.append(chunk)
                active_chunk_index = len(chunks) - 1
                pending_context = []
                if len(chunks) >= max_results:
                    break
                continue
            if event_type == "end":
                pending_context = []
                active_chunk_index = None

        return chunks

    def _line_event(self, data: dict[str, object]) -> tuple[int, str] | None:
        line_number = data.get("line_number")
        lines = data.get("lines", {})
        if not isinstance(line_number, int) or not isinstance(lines, dict):
            return None
        text = lines.get("text", "")
        return line_number, str(text).rstrip("\n")

    def _relative_path(self, data: dict[str, object], repo_path: Path) -> str:
        path_data = data.get("path", {})
        path_text = path_data.get("text", "") if isinstance(path_data, dict) else ""
        path = Path(str(path_text))
        try:
            return path.relative_to(repo_path).as_posix()
        except ValueError:
            return path.as_posix()

    def _chunk(self, file_path: str, lines: list[tuple[int, str]]) -> RetrievalChunk:
        line_start = min(line_number for line_number, _ in lines)
        line_end = max(line_number for line_number, _ in lines)
        content = "\n".join(text for _, text in sorted(lines))
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ripgrep:{file_path}:{line_start}:{line_end}:{content}"))
        return RetrievalChunk(
            chunk_id=chunk_id,
            source="ripgrep",
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            content=content,
            retrieval_method="ripgrep",
        )

    def _append_context(self, chunk: RetrievalChunk, line: tuple[int, str]) -> RetrievalChunk:
        line_number, text = line
        content = f"{chunk.content}\n{text}" if chunk.content else text
        line_start = min(chunk.line_start, line_number)
        line_end = max(chunk.line_end, line_number)
        return RetrievalChunk(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            file_path=chunk.file_path,
            line_start=line_start,
            line_end=line_end,
            content=content,
            retrieval_method=chunk.retrieval_method,
        )

