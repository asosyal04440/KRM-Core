from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.training.eval_tiny_core import evaluate_tiny_core, write_eval_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a tiny KRM-native model artifact if present.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    report = evaluate_tiny_core(args.model, args.tokenizer, args.corpus)
    if report.get("ok"):
        write_eval_report(report, args.model / "eval_report.json")
    if args.json_output:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0 if report.get("ok") else 2
    print("KRM-Core tiny model evaluation")
    print(f"ok: {report.get('ok')}")
    if not report.get("ok"):
        print(f"error: {report.get('error')}")
        return 2
    print(f"checkpoint_found: {report['checkpoint_found']}")
    print(f"config_found: {report['config_found']}")
    print(f"records_checked: {report['record_count_checked']}")
    print(f"notes: {report['notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
