from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.pipeline import inspect_shard


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a KRM-Core V0 shard.")
    parser.add_argument("--mind", required=True, type=Path)
    parser.add_argument("--shard", default="history")
    args = parser.parse_args()
    result = inspect_shard(args.mind, args.shard)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
