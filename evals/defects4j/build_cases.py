from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import yaml


DEFAULT_PROJECTS = ["Lang", "Math", "Time", "Closure"]
DEFAULT_SOURCE_ROOTS = {
    "Closure": "src",
}
CURATED_CASES = {
    "Lang_1b": {
        "query": "NumberFormatException createNumber 80000000 hexadecimal parsing",
        "impact_symbol": "createInteger",
        "expected_methods": [
            "org.apache.commons.lang3.math.NumberUtils.createNumber",
            "org.apache.commons.lang3.math.NumberUtils.createInteger",
        ],
    }
}


@dataclass(frozen=True)
class Defects4JBugMetadata:
    project: str
    bug_id: str
    modified_classes: list[str]
    triggering_tests: list[str]

    @property
    def case_id(self) -> str:
        return f"{self.project}_{self.bug_id}b"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Defects4J localization benchmark cases.")
    parser.add_argument("--defects4j-home", type=Path, required=True)
    parser.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    parser.add_argument("--limit-per-project", type=int, help="Limit generated bugs per project.")
    parser.add_argument("--output", type=Path, default=Path("evals/defects4j/benchmark_cases.yaml"))
    parser.add_argument(
        "--resolve-layout",
        action="store_true",
        help="Checkout each bug and export its source root for more accurate expected file paths.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/private/tmp/codeatlas-defects4j-case-build"),
        help="Temporary checkout directory used with --resolve-layout.",
    )
    args = parser.parse_args()

    cases = build_cases(
        args.defects4j_home,
        projects=args.projects,
        limit_per_project=args.limit_per_project,
        resolve_layout=args.resolve_layout,
        work_dir=args.work_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")


def build_cases(
    defects4j_home: Path,
    projects: Iterable[str] = DEFAULT_PROJECTS,
    limit_per_project: int | None = None,
    resolve_layout: bool = False,
    work_dir: Path | None = None,
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    layout_cache: dict[str, str] = {}
    for project in projects:
        bugs = query_project_bugs(defects4j_home, project)
        if limit_per_project is not None:
            bugs = bugs[:limit_per_project]
        for bug in bugs:
            source_root = source_root_for_bug(
                defects4j_home,
                bug,
                resolve_layout=resolve_layout,
                work_dir=work_dir,
                layout_cache=layout_cache,
            )
            cases.append(case_from_bug(bug, source_root))
    return cases


def query_project_bugs(defects4j_home: Path, project: str) -> list[Defects4JBugMetadata]:
    output = run_defects4j(
        defects4j_home,
        ["query", "-p", project, "-q", "bug.id,classes.modified,tests.trigger"],
    )
    bugs: list[Defects4JBugMetadata] = []
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        bug_id = row[0].strip()
        if not bug_id.isdigit():
            continue
        modified_classes = split_metadata_list(row[1] if len(row) > 1 else "")
        triggering_tests = split_metadata_list(row[2] if len(row) > 2 else "")
        if not modified_classes:
            continue
        bugs.append(
            Defects4JBugMetadata(
                project=project,
                bug_id=bug_id,
                modified_classes=modified_classes,
                triggering_tests=triggering_tests,
            )
        )
    return bugs


def case_from_bug(bug: Defects4JBugMetadata, source_root: str) -> dict[str, object]:
    expected_files = sorted(
        {
            class_name_to_file_path(class_name, source_root)
            for class_name in bug.modified_classes
        }
    )
    modified_simple_names = [simple_name(class_name) for class_name in bug.modified_classes]
    trigger_terms = trigger_query_terms(bug.triggering_tests)
    curated = CURATED_CASES.get(bug.case_id, {})

    return {
        "bug_id": bug.case_id,
        "project": bug.project,
        "query": curated.get("query") or build_query(modified_simple_names, trigger_terms),
        "impact_symbol": curated.get("impact_symbol") or modified_simple_names[0],
        "expected_files": expected_files,
        "expected_methods": curated.get("expected_methods", []),
        "modified_classes": bug.modified_classes,
        "triggering_tests": bug.triggering_tests,
    }


def source_root_for_bug(
    defects4j_home: Path,
    bug: Defects4JBugMetadata,
    resolve_layout: bool,
    work_dir: Path | None,
    layout_cache: dict[str, str],
) -> str:
    if not resolve_layout:
        return default_source_root(bug.project)
    if bug.case_id in layout_cache:
        return layout_cache[bug.case_id]

    if work_dir is None:
        with TemporaryDirectory() as tmpdir:
            root = resolve_source_root(defects4j_home, bug, Path(tmpdir) / bug.case_id)
    else:
        checkout_dir = work_dir / bug.case_id
        root = resolve_source_root(defects4j_home, bug, checkout_dir)
    layout_cache[bug.case_id] = root
    return root


def resolve_source_root(defects4j_home: Path, bug: Defects4JBugMetadata, checkout_dir: Path) -> str:
    if checkout_dir.exists():
        shutil.rmtree(checkout_dir)
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    run_defects4j(
        defects4j_home,
        ["checkout", "-p", bug.project, "-v", f"{bug.bug_id}b", "-w", str(checkout_dir)],
    )
    root = run_defects4j(
        defects4j_home,
        ["export", "-p", "dir.src.classes"],
        cwd=checkout_dir,
    ).strip()
    return root or default_source_root(bug.project)


def run_defects4j(defects4j_home: Path, args: list[str], cwd: Path | None = None) -> str:
    command = [str(defects4j_home / "framework/bin/defects4j"), *args]
    result = subprocess.run(
        command,
        cwd=cwd or defects4j_home,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def class_name_to_file_path(class_name: str, source_root: str) -> str:
    normalized_root = source_root.strip("/")
    relative = f"{class_name.replace('.', '/')}.java"
    return f"{normalized_root}/{relative}" if normalized_root else relative


def default_source_root(project: str) -> str:
    return DEFAULT_SOURCE_ROOTS.get(project, "src/main/java")


def split_metadata_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def build_query(modified_simple_names: list[str], trigger_terms: list[str]) -> str:
    terms = [*modified_simple_names, *trigger_terms]
    deduped = list(dict.fromkeys(term for term in terms if term))
    return " ".join(deduped[:8])


def trigger_query_terms(triggering_tests: list[str]) -> list[str]:
    terms: list[str] = []
    for test in triggering_tests[:4]:
        test_class, _, test_method = test.partition("::")
        terms.append(simple_name(test_class))
        if test_method:
            terms.append(test_method)
    return terms


def simple_name(qualified_name: str) -> str:
    name = qualified_name.rsplit(".", 1)[-1]
    if "$" in name:
        return name.rsplit("$", 1)[-1]
    return name


if __name__ == "__main__":
    main()
