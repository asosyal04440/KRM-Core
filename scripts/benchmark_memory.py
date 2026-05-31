from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.evals.benchmark import DEMO_QUERIES
from krm.pipeline import run_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark KRM-Core V0 estimated memory and latency.")
    parser.add_argument("--mind", required=True, type=Path)
    parser.add_argument("--profile", default="local_core")
    args = parser.parse_args()
    print("KRM-Core memory benchmark")
    for query in DEMO_QUERIES:
        result = run_query(args.mind, query, args.profile)
        print(f"- query: {query}")
        print(f"  estimated_ram_bytes: {result['estimated_ram_bytes']}")
        print(f"  selected_shards: {len(result['selected_shards'])}")
        print(f"  candidate_concepts: {result['candidate_concept_count']}")
        print(f"  ghost_edges: {result['ghost_edge_count']}")
        print(f"  latency_ms: {result['timing_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
