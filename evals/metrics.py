from __future__ import annotations

from dataclasses import dataclass

from codeatlas.storage.models import SearchHit


@dataclass(frozen=True)
class CaseMetrics:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    hit_at_1: float
    context_compression_ratio: float


def evaluate_case(
    hits: list[SearchHit],
    expected_symbols: list[str],
    expected_files: list[str],
    raw_context_tokens: int,
) -> CaseMetrics:
    relevant = [*expected_symbols, *expected_files]
    if not relevant:
        relevant = expected_files

    return CaseMetrics(
        recall_at_5=recall_at_k(hits, relevant, 5),
        recall_at_10=recall_at_k(hits, relevant, 10),
        mrr=mrr(hits, relevant),
        hit_at_1=hit_at_1(hits, relevant),
        context_compression_ratio=compression_ratio(raw_context_tokens, retrieved_context_tokens(hits)),
    )


def recall_at_k(hits: list[SearchHit], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_hits = hits[:k]
    found = {item for item in relevant if any(_matches(hit, item) for hit in top_hits)}
    return len(found) / len(set(relevant))


def mrr(hits: list[SearchHit], relevant: list[str]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if any(_matches(hit, item) for item in relevant):
            return 1.0 / rank
    return 0.0


def hit_at_1(hits: list[SearchHit], relevant: list[str]) -> float:
    if not hits:
        return 0.0
    return 1.0 if any(_matches(hits[0], item) for item in relevant) else 0.0


def retrieved_context_tokens(hits: list[SearchHit]) -> int:
    return sum(_token_count(hit.content) for hit in hits)


def compression_ratio(raw_context_tokens: int, retrieved_tokens: int) -> float:
    if retrieved_tokens <= 0:
        return 0.0
    return raw_context_tokens / retrieved_tokens


def _matches(hit: SearchHit, expected: str) -> bool:
    return expected == hit.symbol or expected == hit.file_path or expected in hit.content


def _token_count(text: str) -> int:
    return len(text.split())

