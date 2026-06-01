from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


ARCH_RWKV = "rwkv"
ARCH_TRANSFORMER = "transformer"


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
    arch: str = ARCH_TRANSFORMER
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
    if config.arch == ARCH_RWKV:
        per_layer_time_mix = 4 * config.d_model * config.d_model  # K, V, R, output linear
        per_layer_channel_mix = 2 * config.d_model * config.d_ff + config.d_model * config.d_model  # K, V, R
        per_layer_wkv = config.d_model * 2  # log_gain, log_decay
        per_layer_norms = 4 * config.d_model
        per_layer = per_layer_time_mix + per_layer_channel_mix + per_layer_wkv + per_layer_norms
        blocks = config.n_layers * per_layer
    else:
        per_layer_attention = 4 * config.d_model * config.d_model
        per_layer_ff = 2 * config.d_model * config.d_ff
        per_layer_norms = 4 * config.d_model
        blocks = config.n_layers * (per_layer_attention + per_layer_ff + per_layer_norms)
    head = config.d_model * config.vocab_size
    return int(embeddings + blocks + head)


def validate_config(config: TinyModelConfig) -> None:
    if config.arch == ARCH_TRANSFORMER and config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if config.max_seq_len <= 0 or config.vocab_size <= 0:
        raise ValueError("vocab_size and max_seq_len must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def get_model_config(name: str, vocab_size: int = 274) -> TinyModelConfig:
    key = name.lower()
    if key in {"10m", "krm_native_10m"}:
        return TinyModelConfig("KRM_NATIVE_10M", vocab_size, 512, 6, 4, 384, 1024, 0.05, "transformer", "tiny")
    if key in {"30m", "krm_native_30m"}:
        return TinyModelConfig("KRM_NATIVE_30M", vocab_size, 1024, 8, 8, 640, 1792, 0.05, "transformer", "small")
    if key in {"100m", "krm_native_100m"}:
        return TinyModelConfig("KRM_NATIVE_100M", vocab_size, 1024, 12, 8, 896, 3584, 0.05, "transformer", "small")
    if key in {"rwkv_10m", "krm_rwkv_10m"}:
        return TinyModelConfig("KRM_RWKV_10M", vocab_size, 512, 6, 1, 384, 1536, 0.05, "rwkv", "tiny")
    if key in {"rwkv_50m", "krm_rwkv_50m"}:
        return TinyModelConfig("KRM_RWKV_50M", vocab_size, 1024, 12, 1, 640, 2560, 0.1, "rwkv", "small")
    if key in {"rwkv_200m", "krm_rwkv_200m"}:
        return TinyModelConfig("KRM_RWKV_200M", vocab_size, 2048, 16, 1, 1024, 4096, 0.1, "rwkv", "small")
    if key in {"rwkv_1b", "krm_rwkv_1b"}:
        return TinyModelConfig("KRM_RWKV_1B", vocab_size, 4096, 24, 1, 1536, 6144, 0.1, "rwkv", "full")
    if key in {"rwkv_7b", "krm_rwkv_7b"}:
        return TinyModelConfig("KRM_RWKV_7B", vocab_size, 8192, 32, 1, 4096, 16384, 0.1, "rwkv", "full")
    raise ValueError(f"unknown model config: {name}")
