from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.evals.benchmark import DEMO_QUERIES
from krm.evals.rubric import evaluate_demo_answer
from krm.pipeline import run_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic KRM-Core demo quality checks.")
    parser.add_argument("--mind", required=True, type=Path)
    parser.add_argument("--profile", default="local_core")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    results = []
    for query in DEMO_QUERIES:
        run = run_query(args.mind, query, args.profile)
        rubric = evaluate_demo_answer(query, run["final_answer"], run["answer_plan"]).to_dict()
        results.append(rubric)
    if args.json_output:
        print(json.dumps(results, indent=2, ensure_ascii=True))
    else:
        for item in results:
            print(f"query: {item['query']}")
            print(f"score: {item['score']} pass={item['passed']}")
            print(f"missing concepts: {', '.join(item['missing_concepts']) or 'none'}")
            print(f"missing structure: {', '.join(item['missing_structure']) or 'none'}")
            print(f"recommendations: {', '.join(item['recommendations']) or 'none'}")
            print()
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
