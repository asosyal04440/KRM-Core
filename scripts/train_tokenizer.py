from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.training.corpus_builder import formatted_corpus_text, read_corpus_records
from krm.training.tokenizer import CharByteTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the lightweight KRM byte tokenizer.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=274)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    records = read_corpus_records(args.corpus, "train") + read_corpus_records(args.corpus, "valid") + read_corpus_records(args.corpus, "test")
    tokenizer = CharByteTokenizer()
    sample_text = formatted_corpus_text(records[:20])
    encoded = tokenizer.encode(sample_text)
    result = {"corpus": str(args.corpus), "out": str(args.out), "dry_run": args.dry_run, "record_count": len(records), "vocab_size": tokenizer.vocab_size, "sample_token_count": len(encoded)}
    if not args.dry_run:
        tokenizer.save(args.out)
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print("KRM-Core tokenizer build")
    print(f"corpus: {result['corpus']}")
    print(f"out: {result['out']}")
    print(f"dry run: {result['dry_run']}")
    print(f"records: {result['record_count']}")
    print(f"vocab_size: {result['vocab_size']}")
    print(f"sample_token_count: {result['sample_token_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
