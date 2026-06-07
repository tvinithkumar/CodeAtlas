from __future__ import annotations

from collections.abc import Iterable

from codeatlas.storage.models import SearchHit


class ReciprocalRankFusion:
    def __init__(self, rank_constant: int = 60) -> None:
        self.rank_constant = rank_constant

    def fuse(self, ranked_lists: Iterable[tuple[float, list[SearchHit]]], limit: int = 10) -> list[SearchHit]:
        scores: dict[str, float] = {}
        hits: dict[str, SearchHit] = {}
        for weight, ranked_hits in ranked_lists:
            for rank, hit in enumerate(ranked_hits, start=1):
                key = self._dedupe_key(hit)
                scores[key] = scores.get(key, 0.0) + weight / (self.rank_constant + rank)
                hits.setdefault(key, hit)

        return [
            self._with_score(hits[key], score)
            for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    def _dedupe_key(self, hit: SearchHit) -> str:
        if hit.line_start is not None and hit.line_end is not None:
            return f"{hit.file_path}:{hit.line_start}:{hit.line_end}"
        if hit.symbol:
            return f"{hit.file_path}:{hit.symbol}"
        return hit.id

    def _with_score(self, hit: SearchHit, score: float) -> SearchHit:
        return SearchHit(
            id=hit.id,
            score=score,
            file_path=hit.file_path,
            symbol=hit.symbol,
            content=hit.content,
            source=hit.source,
            retrieval_method=hit.retrieval_method,
            line_start=hit.line_start,
            line_end=hit.line_end,
        )
