from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.pipeline import run_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the KRM-Core V0 query pipeline.")
    parser.add_argument("--mind", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--profile", default="local_core")
    parser.add_argument("--max-concepts", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    result = run_query(
        args.mind,
        args.query,
        args.profile,
        max_concepts=args.max_concepts,
        max_edges=args.max_edges,
        rounds=args.rounds,
    )
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print(f"query: {result['query']}")
    print(f"profile: {result['profile']}")
    print(f"detected intent: {result['intent']}")
    print(f"selected shards: {', '.join(result['selected_shards'])}")
    print(f"candidate concept count: {result['candidate_concept_count']}")
    print(f"ghost edge count: {result['ghost_edge_count']}")
    print(f"resonance rounds: {result['resonance_rounds']}")
    print("top activated concepts:")
    for item in result["top_activated_concepts"][:10]:
        print(f"- {item['name']} ({item['score']:.3f})")
    print("answer plan:")
    print(json.dumps(result["answer_plan"], indent=2, ensure_ascii=True))
    print("final mock answer:")
    print(result["final_answer"])
    print(f"estimated RAM: {result['estimated_ram_bytes']} bytes")
    print(f"timing: {result['timing_ms']} ms")
    if result["degradation_decisions"]:
        print("degradation decisions:")
        for decision in result["degradation_decisions"]:
            print(f"- {decision}")
    rubric = result.get("quality_rubric", {})
    if rubric:
        print(f"quality rubric: score={rubric.get('score')} pass={rubric.get('passed')}")
    if args.trace:
        print("resonance trace:")
        trace = result.get("resonance_trace", {})
        for idx, round_info in enumerate(trace.get("rounds", []), start=1):
            names = ", ".join(item["concept"] for item in round_info.get("top_concepts", [])[:6])
            print(f"- round {idx}: {names}")
        for edge in trace.get("top_edges_used", [])[:8]:
            print(f"- edge {edge['edge_type']}: {edge['src']} -> {edge['dst']} ({edge['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
