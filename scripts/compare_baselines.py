from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.evals.baselines import krm_resonance_answer, lexical_only_baseline, retrieval_without_resonance_baseline
from krm.evals.benchmark import DEMO_QUERIES


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare KRM-Core V0.1 resonance with lightweight baselines.")
    parser.add_argument("--mind", required=True, type=Path)
    parser.add_argument("--profile", default="local_core")
    args = parser.parse_args()
    for query in DEMO_QUERIES:
        lexical = lexical_only_baseline(args.mind, query)
        retrieval = retrieval_without_resonance_baseline(args.mind, query, args.profile)
        krm = krm_resonance_answer(args.mind, query, args.profile)
        print(f"query: {query}")
        print(f"lexical-only top concepts: {', '.join(lexical['top_concepts'])}")
        print(f"retrieval-only top concepts: {', '.join(retrieval['top_concepts'])}")
        print(f"KRM resonance top concepts: {', '.join(krm['top_concepts'])}")
        print("answer quality notes: resonance adds query-local edges, answer planning, and grounded structure")
        print(f"estimated RAM: {krm['estimated_ram_bytes']} bytes")
        print(f"timing: lexical={lexical['timing_ms']} ms retrieval={retrieval['timing_ms']} ms krm={krm['timing_ms']} ms")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
