from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.training.example_forge import DEFAULT_TYPES, forge_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Forge lightweight JSONL training examples without training a model.")
    parser.add_argument("--mind", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--types", default=",".join(sorted(DEFAULT_TYPES)))
    parser.add_argument("--include-source", default="dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    types = {item.strip() for item in args.types.split(",") if item.strip()}
    result = forge_examples(args.mind, args.out, args.max_examples, types, args.dry_run)
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print("KRM-Core training example forge")
    print(f"mind: {result['mind']}")
    print(f"out: {result['out']}")
    print(f"dry run: {result['dry_run']}")
    print(f"types: {', '.join(result['types'])}")
    print(f"example_count: {result['example_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
