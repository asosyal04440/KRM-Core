from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.pipeline import ingest_articles
from krm.source.dataset_discovery import DatasetDiscovery, summarize_discovery
from krm.source.zim_reader import LocalFolderSourceReader


def _suffix_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}" for item in raw.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely ingest small local KRM-Core datasets without downloads.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", default="tiny")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--max-articles", type=int, default=10_000)
    parser.add_argument("--max-file-mb", type=float, default=25.0)
    parser.add_argument("--jsonl-title-field", default="title")
    parser.add_argument("--jsonl-text-field", default="text")
    parser.add_argument("--csv-title-field", default="title")
    parser.add_argument("--csv-text-field", default="text")
    parser.add_argument("--include-suffix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    discovered = DatasetDiscovery().scan(args.source, recursive=args.recursive)
    include_suffixes = _suffix_set(args.include_suffix)
    reader = LocalFolderSourceReader(
        args.source,
        recursive=args.recursive,
        max_files=args.max_files,
        max_articles=args.max_articles,
        max_file_mb=args.max_file_mb,
        include_suffixes=include_suffixes,
        jsonl_title_field=args.jsonl_title_field,
        jsonl_text_field=args.jsonl_text_field,
        csv_title_field=args.csv_title_field,
        csv_text_field=args.csv_text_field,
    )
    selected_files = reader.discover()
    articles = list(reader.iter_articles())
    article_count = len(articles)
    warnings = list(reader.warnings)
    if args.dry_run:
        ingest_stats = None
    elif articles:
        ingest_stats = ingest_articles(articles, args.out)
        warnings.extend(ingest_stats.get("warnings", []))
    else:
        ingest_stats = {
            "articles": 0,
            "concepts": 0,
            "warnings": ["No sufficient local concepts found. Add more ingestible local files or lower thresholds."],
        }
        warnings.extend(ingest_stats["warnings"])

    summary = summarize_discovery(discovered)
    selected_size = sum(item.size_bytes for item in selected_files)
    payload = {
        "source": str(args.source),
        "out": str(args.out),
        "profile": args.profile,
        "dry_run": args.dry_run,
        "discovery": summary,
        "selected_files": [item.to_dict() for item in selected_files],
        "selected_file_count": len(selected_files),
        "selected_size_bytes": selected_size,
        "estimated_artifact_size_bytes": max(4096, selected_size // 8) if selected_files else 0,
        "estimated_ram_impact_bytes": max(8 * 1024 * 1024, min(selected_size * 2, 512 * 1024 * 1024)) if selected_files else 0,
        "articles": article_count if args.dry_run else (ingest_stats or {}).get("articles", 0),
        "concepts": 0 if args.dry_run else (ingest_stats or {}).get("concepts", 0),
        "warnings": sorted(set(warnings + summary["warnings"])),
    }
    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    print("KRM-Core local ingest")
    print(f"source: {args.source}")
    print(f"out: {args.out}")
    print(f"profile: {args.profile}")
    print(f"dry run: {args.dry_run}")
    print(f"total files scanned: {summary['total_files_scanned']}")
    print(f"selected ingestible files: {payload['selected_file_count']}")
    print(f"skipped files: {summary['skipped_files']}")
    print(f"selected size: {round(selected_size / (1024 * 1024), 3)} MB")
    print(f"estimated artifact size: {payload['estimated_artifact_size_bytes']} bytes")
    print(f"estimated RAM impact: {payload['estimated_ram_impact_bytes']} bytes")
    print(f"articles: {payload['articles']}")
    print(f"concepts: {payload['concepts']}")
    for warning in payload["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
