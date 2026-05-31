from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.training.corpus_builder import build_training_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KRM-native training corpus JSONL shards from local artifacts.")
    parser.add_argument("--mind", type=Path)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", default="tiny", choices=["tiny", "small", "router_only", "planner_only", "composer_only"])
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-input-chars", type=int, default=20000)
    parser.add_argument("--max-target-chars", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tasks = {item.strip().upper() for item in args.tasks.split(",") if item.strip()}
    result = build_training_corpus(args.mind, args.examples, args.out, args.profile, tasks, args.max_records, args.max_input_chars, args.max_target_chars, args.dry_run, args.seed)
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print("KRM-Core training corpus build")
    print(f"out: {result['out']}")
    print(f"dry run: {result['dry_run']}")
    print(f"records: {result['record_count']}")
    print(f"splits: {result['splits']}")
    print(f"tasks: {result['manifest']['task_distribution']}")
    print(f"warnings: {result['manifest']['warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
