"""Download and preprocess HuggingFace datasets for RWKV training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HF_DATASETS = {
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT"),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "the_stack_v2": ("bigcode/the-stack-v2", None),
    "c4": ("allenai/c4", "en"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download & preprocess HF datasets for RWKV training")
    parser.add_argument("--dataset", choices=list(HF_DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    from krm.training.bpe_tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer)
    result = {}

    if args.dry_run:
        result = {
            "dry_run": True,
            "dataset": args.dataset,
            "out": str(args.out),
            "max_samples": args.max_samples,
            "tokenizer_vocab": tokenizer.vocab_size,
            "seq_len": args.seq_len,
        }
        return _print(result, args.json_output)

    try:
        from datasets import load_dataset, Dataset
    except ImportError:
        result = {"error": "datasets not installed. Run: pip install datasets"}
        return _print(result, args.json_output)

    datasets_to_process = list(HF_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    args.out.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for ds_name in datasets_to_process:
        path, config = HF_DATASETS[ds_name]
        print(f"Loading {ds_name} ({path})...")
        try:
            ds = load_dataset(path, config, split="train", streaming=True)
        except Exception as e:
            print(f"  Error loading {ds_name}: {e}")
            all_stats[ds_name] = {"status": "error", "message": str(e)}
            continue

        count = 0
        token_count = 0
        output_file = args.out / f"{ds_name}_tokenized.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for i, example in enumerate(ds):
                if i >= args.max_samples:
                    break
                text = example.get("text", "")
                if not text:
                    continue
                ids = tokenizer.encode(text)
                for start in range(0, len(ids), args.seq_len):
                    chunk = ids[start:start + args.seq_len]
                    if len(chunk) < args.seq_len // 2:
                        continue
                    f.write(json.dumps({"tokens": chunk}) + "\n")
                    token_count += len(chunk)
                    count += 1
                    if count % 1000 == 0:
                        print(f"  {ds_name}: {count} chunks, {token_count} tokens")

        stats = {
            "status": "ok",
            "chunks": count,
            "tokens": token_count,
            "file": str(output_file),
        }
        all_stats[ds_name] = stats
        print(f"  {ds_name} done: {count} chunks, {token_count} tokens")

    result = {
        "dry_run": False,
        "out": str(args.out),
        "datasets": all_stats,
    }
    return _print(result, args.json_output)


def _print(result: dict, json_output: bool) -> int:
    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print("HF Dataset Preparation")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
