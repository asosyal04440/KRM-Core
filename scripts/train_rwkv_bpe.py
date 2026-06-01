"""Train RWKV model using BPE tokenizer on real training data."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RWKV model with BPE tokenizer")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", default="rwkv_10m")
    parser.add_argument("--corpus", type=Path, help="Path to training corpus (txt file)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    from krm.training.bpe_tokenizer import BPETokenizer
    from krm.training.model_config import get_model_config
    from krm.training.torch_backend import build_model, train_step, save_checkpoint

    tokenizer = BPETokenizer.load(args.tokenizer)
    config = get_model_config(args.config, vocab_size=tokenizer.vocab_size)

    plan = {
        "tokenizer": str(args.tokenizer),
        "config": config.to_dict(),
        "out": str(args.out),
        "dry_run": args.dry_run,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "max_steps": args.max_steps,
    }

    if args.dry_run:
        return _print(plan, args.json_output)

    import torch

    # Load and tokenize corpus
    if args.corpus:
        text = args.corpus.read_text(encoding="utf-8")
    else:
        print("No corpus specified, using hardcoded test text")
        text = "This is a test of the KRM RWKV training pipeline with BPE tokenization. " * 100

    print(f"Corpus size: {len(text)} chars, ~{len(text)//4} tokens estimated")
    ids = tokenizer.encode(text)
    print(f"Tokenized: {len(ids)} tokens")

    # Prepare tensors
    if len(ids) < args.seq_len + 1:
        ids = ids * ((args.seq_len + 1) // len(ids) + 1)
    tensor = torch.tensor(ids[: args.batch_size * (args.seq_len + 1)], dtype=torch.long)
    chunks = tensor.unfold(0, args.seq_len + 1, args.seq_len + 1)
    batch = chunks[: args.batch_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model: {config.model_name}, params={config.parameter_estimate:,}")

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out / "tokenizer")
    config.save(args.out / "config.json")

    losses = []
    batch = batch.to(device)

    for step in range(1, args.max_steps + 1):
        loss = train_step(model, batch, optimizer)
        losses.append(loss)
        if step % 10 == 0:
            avg = sum(losses[-10:]) / min(len(losses), 10)
            print(f"Step {step}/{args.max_steps}: loss={loss:.4f}, avg10={avg:.4f}")
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(model, config, args.out / f"checkpoint_step_{step}.pt", step)

    metadata = {
        **plan,
        "dry_run": False,
        "device": str(device),
        "final_loss": losses[-1] if losses else None,
        "avg_loss_last_10": sum(losses[-10:]) / min(len(losses), 10) if losses else None,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    (args.out / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return _print(metadata, args.json_output)


def _print(result: dict, json_output: bool) -> int:
    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print("RWKV-BPE Training")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
