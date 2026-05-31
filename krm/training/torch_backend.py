from __future__ import annotations

from pathlib import Path
from typing import Any

from krm.training.model_config import TinyModelConfig


def torch_availability() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return {
            "available": False,
            "message": "Optional PyTorch backend is not installed. Install manually with `pip install -e .[train]` if you want tiny smoke training.",
        }
    return {"available": True, "message": f"PyTorch available: {torch.__version__}"}


def build_tiny_decoder_lm(config: TinyModelConfig):
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    class TinyDecoderLM(nn.Module):
        def __init__(self, cfg: TinyModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        def forward(self, input_ids):
            batch, seq = input_ids.shape
            positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, seq)
            x = self.token(input_ids) + self.pos(positions)
            mask = torch.triu(torch.ones(seq, seq, device=input_ids.device, dtype=torch.bool), diagonal=1)
            x = self.blocks(x, mask=mask)
            return self.head(self.norm(x))

    return TinyDecoderLM(config)


def train_step(model, input_ids, optimizer) -> float:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def save_checkpoint(model, config: TinyModelConfig, path: Path, step: int) -> None:
    import torch  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.to_dict(), "step": step}, path)


def load_checkpoint(path: Path, config: TinyModelConfig):
    import torch  # type: ignore

    model = build_tiny_decoder_lm(config)
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    return model, int(payload.get("step", 0))


def generate_greedy(model, input_ids, max_new_tokens: int = 32):
    import torch  # type: ignore

    model.eval()
    with torch.no_grad():
        current = input_ids
        for _ in range(max_new_tokens):
            logits = model(current)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current = torch.cat([current, next_id], dim=1)
    return current
