from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from krm.training.resource_estimator import estimate_by_name


def _mb(value: int) -> float:
    return round(value / 1024 / 1024, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate KRM-native tiny training resources.")
    parser.add_argument("--config", default="10m", choices=["10m", "30m", "100m"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=274)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    result = estimate_by_name(args.config, args.batch_size, args.seq_len, args.vocab_size).to_dict()
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    print("KRM-Core training resource estimate")
    print(f"config: {args.config}")
    print(f"parameters: {result['parameter_count']}")
    print(f"model fp32 MB: {_mb(result['model_memory_fp32_bytes'])}")
    print(f"model fp16 MB: {_mb(result['model_memory_fp16_bytes'])}")
    print(f"optimizer MB: {_mb(result['optimizer_memory_bytes'])}")
    print(f"activations MB: {_mb(result['activation_memory_bytes'])}")
    print(f"checkpoint MB: {_mb(result['checkpoint_size_bytes'])}")
    print(f"total training MB: {_mb(result['total_training_memory_bytes'])}")
    print(f"warnings: {result['warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
