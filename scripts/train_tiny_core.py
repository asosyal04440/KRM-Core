from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.training.corpus_builder import formatted_corpus_text, read_corpus_records
from krm.training.model_config import get_model_config
from krm.training.resource_estimator import estimate_training_resources
from krm.training.tokenizer import CharByteTokenizer
from krm.training.torch_backend import build_model, save_checkpoint, torch_availability, train_step


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional tiny KRM-native smoke training.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default="10m", choices=["10m", "30m", "100m", "rwkv_10m", "rwkv_50m", "rwkv_200m"])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    tokenizer = CharByteTokenizer.load(args.tokenizer)
    config = get_model_config(args.config, vocab_size=tokenizer.vocab_size)
    estimate = estimate_training_resources(config, args.batch_size, args.seq_len)
    records = read_corpus_records(args.corpus, "train")
    availability = torch_availability()
    plan = {
        "corpus": str(args.corpus),
        "tokenizer": str(args.tokenizer),
        "out": str(args.out),
        "config": config.to_dict(),
        "dry_run": args.dry_run,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "record_count": len(records),
        "resource_estimate": estimate.to_dict(),
        "torch": availability,
    }
    if args.dry_run:
        return _print(plan, args.json_output)
    if not availability["available"]:
        plan["error"] = availability["message"]
        _print(plan, args.json_output)
        return 2
    if not records:
        plan["error"] = "no training records found"
        _print(plan, args.json_output)
        return 2

    import torch  # type: ignore

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    text = formatted_corpus_text(records)
    ids = tokenizer.encode(text)
    if len(ids) < args.seq_len + 1:
        ids = ids + [tokenizer.token_to_id["<|END|>"]] * (args.seq_len + 1 - len(ids))
    tensor = torch.tensor(ids[: max(args.seq_len + 1, args.batch_size * (args.seq_len + 1))], dtype=torch.long)
    chunks = tensor.unfold(0, args.seq_len + 1, args.seq_len + 1)
    batch = chunks[: args.batch_size].to(device)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    for step in range(1, args.max_steps + 1):
        losses.append(train_step(model, batch, optimizer))
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(model, config, args.out / f"checkpoint_step_{step}.pt", step)
    args.out.mkdir(parents=True, exist_ok=True)
    config.save(args.out / "config.json")
    shutil.copy2(args.tokenizer / "tokenizer.json", args.out / "tokenizer.json")
    metadata = {**plan, "dry_run": False, "device": device, "losses": losses[-10:], "completed_at": datetime.now(UTC).isoformat()}
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    return _print(metadata, args.json_output)


def _print(result: dict, json_output: bool) -> int:
    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print("KRM-Core tiny training plan")
    print(f"out: {result['out']}")
    print(f"dry run: {result['dry_run']}")
    print(f"records: {result['record_count']}")
    print(f"config: {result['config']['model_name']} params={result['config']['parameter_estimate']}")
    print(f"torch: {result['torch']['message']}")
    print(f"estimated training bytes: {result['resource_estimate']['total_training_memory_bytes']}")
    if "error" in result:
        print(f"error: {result['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
