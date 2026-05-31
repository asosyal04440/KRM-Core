from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.evals.dataset_quality import dataset_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Report deterministic local dataset quality signals.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--sample-rows", type=int, default=100)
    args = parser.parse_args()
    report = dataset_quality_report(args.source, sample_rows=args.sample_rows)
    if args.json_output:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0
    print(f"source: {report['source']}")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")
    for item in report["files"]:
        print(f"- {item['path']} ingestible={item['ingestible']}")
        if "schema" in item:
            print(f"  task_type: {item['schema']['task_type']} confidence={item['schema']['confidence']}")
            print(f"  rows_sampled: {item['rows_sampled']}")
            print(f"  empty_row_rate: {item['empty_row_rate']}")
            print(f"  duplicate_row_estimate: {item['duplicate_row_estimate']}")
            print(f"  average_text_length: {item['average_text_length']}")
            print(f"  recommended overrides: {', '.join(item['recommended_mapping_overrides']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
