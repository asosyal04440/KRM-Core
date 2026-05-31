from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.pipeline import ingest_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest plain text or markdown sources into KRM-Core V0 artifacts.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    stats = ingest_source(args.source, args.out)
    print("KRM-Core ingest complete")
    print(f"source: {args.source}")
    print(f"out: {args.out}")
    print(f"articles: {stats['articles']}")
    print(f"concepts: {stats['concepts']}")
    print(json.dumps(stats, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
