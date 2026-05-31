from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TinyModelConfig:
    model_name: str
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float
    task_mix_profile: str = "tiny"
    parameter_estimate: int = 0

    def __post_init__(self) -> None:
        if self.parameter_estimate <= 0:
            self.parameter_estimate = estimate_parameter_count(self)
        validate_config(self)

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TinyModelConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def estimate_parameter_count(config: TinyModelConfig) -> int:
    embeddings = config.vocab_size * config.d_model
    per_layer_attention = 4 * config.d_model * config.d_model
    per_layer_ff = 2 * config.d_model * config.d_ff
    per_layer_norms = 4 * config.d_model
    blocks = config.n_layers * (per_layer_attention + per_layer_ff + per_layer_norms)
    head = config.d_model * config.vocab_size
    return int(embeddings + blocks + head)


def validate_config(config: TinyModelConfig) -> None:
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if config.max_seq_len <= 0 or config.vocab_size <= 0:
        raise ValueError("vocab_size and max_seq_len must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def get_model_config(name: str, vocab_size: int = 274) -> TinyModelConfig:
    key = name.lower()
    if key in {"10m", "krm_native_10m"}:
        return TinyModelConfig("KRM_NATIVE_10M", vocab_size, 512, 6, 4, 384, 1024, 0.05, "tiny")
    if key in {"30m", "krm_native_30m"}:
        return TinyModelConfig("KRM_NATIVE_30M", vocab_size, 1024, 8, 8, 640, 1792, 0.05, "small")
    if key in {"100m", "krm_native_100m"}:
        return TinyModelConfig("KRM_NATIVE_100M", vocab_size, 1024, 12, 8, 896, 3584, 0.05, "small")
    raise ValueError(f"unknown model config: {name}")
