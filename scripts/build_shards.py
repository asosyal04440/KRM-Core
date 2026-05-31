from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.pipeline import build_shards


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KRM-Core V0 shards from concept skeletons.")
    parser.add_argument("--mind", required=True, type=Path)
    parser.add_argument("--profile", default="local_core")
    args = parser.parse_args()
    result = build_shards(args.mind, args.profile)
    print("KRM-Core shard build complete")
    print(f"mind: {args.mind}")
    print(f"profile: {result['profile']}")
    print(f"concepts: {result['concept_count']}")
    print(f"shards: {len(result['shards'])}")
    for shard in result["shards"]:
        print(f"- {shard['shard_id']}: concepts={shard['concept_count']} estimated_ram={shard['estimated_ram_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
