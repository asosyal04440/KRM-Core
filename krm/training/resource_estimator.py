from __future__ import annotations

from dataclasses import dataclass

from krm.training.model_config import TinyModelConfig, get_model_config


@dataclass(slots=True)
class TrainingResourceEstimate:
    parameter_count: int
    model_memory_fp32_bytes: int
    model_memory_fp16_bytes: int
    optimizer_memory_bytes: int
    activation_memory_bytes: int
    checkpoint_size_bytes: int
    total_training_memory_bytes: int
    warnings: list[str]

    def to_dict(self) -> dict[str, int | list[str]]:
        return {
            "parameter_count": self.parameter_count,
            "model_memory_fp32_bytes": self.model_memory_fp32_bytes,
            "model_memory_fp16_bytes": self.model_memory_fp16_bytes,
            "optimizer_memory_bytes": self.optimizer_memory_bytes,
            "activation_memory_bytes": self.activation_memory_bytes,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "total_training_memory_bytes": self.total_training_memory_bytes,
            "warnings": self.warnings,
        }


def estimate_training_resources(config: TinyModelConfig, batch_size: int, seq_len: int) -> TrainingResourceEstimate:
    params = config.parameter_estimate
    fp32 = params * 4
    fp16 = params * 2
    optimizer = params * 8
    effective_seq = min(seq_len, config.max_seq_len)
    activations = batch_size * effective_seq * config.d_model * config.n_layers * 4
    checkpoint = fp32 + 4096
    total = fp32 + optimizer + activations
    warnings: list[str] = []
    if seq_len > config.max_seq_len:
        warnings.append(f"seq_len clipped by model max_seq_len={config.max_seq_len}")
    if total > 6 * 1024**3:
        warnings.append("rough estimate exceeds 6 GB; use smaller batch/seq or CPU/offload")
    if config.model_name == "KRM_NATIVE_100M":
        warnings.append("100M config is roadmap/config-only; do not train by default")
    return TrainingResourceEstimate(params, fp32, fp16, optimizer, activations, checkpoint, total, warnings)


def estimate_by_name(config_name: str, batch_size: int, seq_len: int, vocab_size: int = 274) -> TrainingResourceEstimate:
    return estimate_training_resources(get_model_config(config_name, vocab_size=vocab_size), batch_size, seq_len)
