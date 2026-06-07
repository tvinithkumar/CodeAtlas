from __future__ import annotations

from pathlib import Path

from codeatlas.retrieval.code_window import CodeWindowFetcher
from codeatlas.storage.models import SearchHit
from evals.metrics import evaluate_case


def test_code_window_fetcher_bounds_result_to_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    window = CodeWindowFetcher().get_code_window(repo, "app.py", line=3, radius=1)

    assert window.line_start == 2
    assert window.line_end == 4
    assert window.content == "two\nthree\nfour"


def test_retrieval_metrics_compute_recall_mrr_and_compression() -> None:
    hits = [
        SearchHit(
            id="1",
            score=1.0,
            file_path="service.py",
            symbol="retry_delay",
            content="return RETRY_BACKOFF_MS",
        )
    ]

    metrics = evaluate_case(
        hits,
        expected_symbols=["retry_delay"],
        expected_files=["service.py"],
        raw_context_tokens=100,
    )

    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.hit_at_1 == 1.0
    assert metrics.context_compression_ratio > 1.0

