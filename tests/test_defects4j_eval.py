from __future__ import annotations

from pathlib import Path

from codeatlas.common.config import Settings
from codeatlas.embedding.hash_provider import HashEmbeddingProvider
from codeatlas.indexing.repository_indexer import RepositoryIndexer
from codeatlas.storage.sqlite_store import SQLiteStore
from evals.defects4j.build_cases import (
    Defects4JBugMetadata,
    case_from_bug,
    class_name_to_file_path,
    default_source_root,
    split_metadata_list,
)
from evals.defects4j.benchmark import group_cases_by_bug, parse_bug_id, settings_for_bug
from evals.defects4j.run_fault_localization_eval import _summary, check_qdrant_available, evaluate_localization_case


def test_defects4j_benchmark_groups_cases_and_parses_bug_ids() -> None:
    assert parse_bug_id("Lang_1b") == ("Lang", "1b")
    grouped = group_cases_by_bug(
        [
            {"bug_id": "Lang_1b", "query": "a"},
            {"bug_id": "Lang_1b", "query": "b"},
            {"bug_id": "Math_2b", "query": "c"},
        ]
    )

    assert len(grouped["Lang_1b"]) == 2
    assert len(grouped["Math_2b"]) == 1


def test_defects4j_benchmark_qdrant_collection_suffixes_only_when_indexing(tmp_path: Path) -> None:
    class Args:
        config = None
        with_vectors = True
        reuse_index = False

    settings = settings_for_bug(Args, tmp_path / "Lang_1b.db", "Lang_1b")
    assert settings.qdrant_collection == "codeatlas_chunks_lang_1b"

    Args.reuse_index = True
    settings = settings_for_bug(Args, tmp_path / "Lang_1b.db", "Lang_1b")
    assert settings.qdrant_collection == "codeatlas_chunks"


def test_defects4j_case_builder_maps_modified_classes_to_expected_files() -> None:
    bug = Defects4JBugMetadata(
        project="Lang",
        bug_id="1",
        modified_classes=["org.apache.commons.lang3.math.NumberUtils"],
        triggering_tests=["org.apache.commons.lang3.math.NumberUtilsTest::TestLang747"],
    )

    case = case_from_bug(bug, "src/main/java")

    assert case["bug_id"] == "Lang_1b"
    assert case["query"] == "NumberFormatException createNumber 80000000 hexadecimal parsing"
    assert case["impact_symbol"] == "createInteger"
    assert case["expected_files"] == ["src/main/java/org/apache/commons/lang3/math/NumberUtils.java"]
    assert case["expected_methods"] == [
        "org.apache.commons.lang3.math.NumberUtils.createNumber",
        "org.apache.commons.lang3.math.NumberUtils.createInteger",
    ]


def test_defects4j_case_builder_uses_project_source_roots_and_metadata_lists() -> None:
    assert default_source_root("Closure") == "src"
    assert default_source_root("Math") == "src/main/java"
    assert class_name_to_file_path("com.google.javascript.jscomp.Compiler", "src") == (
        "src/com/google/javascript/jscomp/Compiler.java"
    )
    assert split_metadata_list("A; B ;") == ["A", "B"]


def test_defects4j_summary_ignores_missing_optional_metrics() -> None:
    summary = _summary(
        [
            {"file_recall_at_5": 1.0, "method_recall_at_10": None},
            {"file_recall_at_5": 0.0, "method_recall_at_10": 1.0},
        ]
    )

    assert summary["file_recall_at_5"] == 0.5
    assert summary["method_recall_at_10"] == 1.0


def test_qdrant_preflight_reports_clear_error() -> None:
    settings = Settings.from_dict({"qdrant_url": "http://127.0.0.1:1"})

    try:
        check_qdrant_available(settings)
    except RuntimeError as exc:
        assert "Qdrant is not reachable" in str(exc)
    else:
        raise AssertionError("Expected Qdrant preflight to fail")


def test_fault_localization_eval_scores_expected_file_and_methods(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "NumberUtils.java").write_text(
        """
package org.example;

public class NumberUtils {
    public int createNumber(String value) {
        return createInteger(value);
    }

    public int createInteger(String value) {
        return Integer.decode(value);
    }
}
""".strip(),
        encoding="utf-8",
    )

    settings = Settings(sqlite_path=tmp_path / "codeatlas.db")
    embedding = HashEmbeddingProvider()
    RepositoryIndexer(settings=settings, enable_qdrant=False, embedding_generator=embedding).index(repo)

    case = {
        "bug_id": "Lang_1b",
        "query": "NumberFormatException createNumber hexadecimal parsing",
        "impact_symbol": "createInteger",
        "expected_files": ["NumberUtils.java"],
        "expected_methods": [
            "org.example.NumberUtils.createNumber",
            "org.example.NumberUtils.createInteger",
        ],
    }

    result = evaluate_localization_case(
        SQLiteStore(settings.sqlite_path),
        embedding,
        settings,
        repo,
        case,
        raw_context_tokens=1000,
        limit=10,
        include_vectors=False,
        window_radius=2,
    )

    assert result["metrics"]["file_recall_at_5"] == 1.0
    assert result["metrics"]["method_recall_at_10"] == 1.0
    assert result["metrics"]["mrr"] > 0.0
    assert result["metrics"]["context_compression_ratio"] > 1.0
    assert result["retrieval_method_counts"]
    assert result["vector_hit_count"] == 0
