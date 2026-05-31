from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.training.export import export_tiny_core


def main() -> int:
    parser = argparse.ArgumentParser(description="Export KRM-native tiny model metadata/artifacts.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    result = export_tiny_core(args.model, args.out)
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0 if result.get("ok") else 2
    print("KRM-Core tiny model export")
    print(f"ok: {result.get('ok')}")
    if not result.get("ok"):
        print(f"error: {result.get('error')}")
        return 2
    print(f"out: {result['out']}")
    print(f"copied_files: {result['copied_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
