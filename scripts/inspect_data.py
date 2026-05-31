from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.source.dataset_discovery import DatasetDiscovery, summarize_discovery


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local KRM-Core ingest candidates without reading full files.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    files = DatasetDiscovery().scan(args.source, recursive=args.recursive)
    if args.max_files is not None:
        files = files[: args.max_files]
    summary = summarize_discovery(files)
    payload = {"source": str(args.source), "summary": summary, "files": [item.to_dict() for item in files]}
    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    print(f"source: {args.source}")
    print(f"total files scanned: {summary['total_files_scanned']}")
    print(f"ingestible files: {summary['ingestible_files']}")
    print(f"skipped files: {summary['skipped_files']}")
    print(f"total size: {summary['total_size_mb']} MB")
    print(f"total ingestible size: {summary['total_ingestible_size_mb']} MB")
    print(f"total non-ingestible size: {summary['total_non_ingestible_size_mb']} MB")
    print(f"estimated artifact size: {summary['estimated_artifact_size_bytes']} bytes")
    print(f"estimated RAM impact: {summary['estimated_ram_impact_bytes']} bytes")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    print("detected files:")
    for item in files:
        print(
            f"- {item.path} suffix={item.suffix or '<none>'} size_mb={item.size_bytes / (1024 * 1024):.3f} "
            f"supported={item.supported} ingestible={item.ingestible} reason={item.reason}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
