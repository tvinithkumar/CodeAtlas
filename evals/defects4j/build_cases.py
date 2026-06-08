from __future__ import annotations

import argparse
from pathlib import Path

import yaml


LANG_1B_CASE = {
    "bug_id": "Lang_1b",
    "query": "NumberFormatException createNumber 80000000 hexadecimal parsing",
    "impact_symbol": "createInteger",
    "expected_files": ["src/main/java/org/apache/commons/lang3/math/NumberUtils.java"],
    "expected_methods": [
        "org.apache.commons.lang3.math.NumberUtils.createNumber",
        "org.apache.commons.lang3.math.NumberUtils.createInteger",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seed Defects4J localization eval cases.")
    parser.add_argument("--output", type=Path, default=Path("evals/defects4j/benchmark_cases.yaml"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump([LANG_1B_CASE], sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
