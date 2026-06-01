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


def build_rwkv_model(config: TinyModelConfig):
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

    class WKVMemory(nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            self.d_model = d_model
            self.log_gain = nn.Parameter(torch.zeros(d_model))
            self.log_decay = nn.Parameter(torch.zeros(d_model))

        def forward(self, values: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, d_model = values.shape
            gain = torch.exp(self.log_gain)
            decay = torch.exp(-torch.exp(self.log_decay))
            outputs = torch.zeros_like(values)
            state = torch.zeros(batch_size, d_model, device=values.device)
            norm = torch.zeros(batch_size, d_model, device=values.device)
            for t in range(seq_len):
                v_t = values[:, t, :]
                k_t = keys[:, t, :] * gain
                state = state * decay + k_t * v_t
                norm = norm * decay + k_t
                outputs[:, t, :] = state / (norm + 1e-8)
            return outputs

    class RWKVBlock(nn.Module):
        def __init__(self, d_model: int, d_ff: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.time_key = nn.Linear(d_model, d_model, bias=False)
            self.time_value = nn.Linear(d_model, d_model, bias=False)
            self.time_receptance = nn.Linear(d_model, d_model, bias=False)
            self.time_output = nn.Linear(d_model, d_model, bias=False)
            self.wkv = WKVMemory(d_model)
            self.channel_key = nn.Linear(d_model, d_ff, bias=False)
            self.channel_value = nn.Linear(d_ff, d_model, bias=False)
            self.channel_receptance = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            h = self.ln1(x)
            k = self.time_key(h)
            v = self.time_value(h)
            r = torch.sigmoid(self.time_receptance(h))
            wkv_out = self.wkv(v, k)
            h = residual + r * self.time_output(wkv_out)
            residual = h
            h = self.ln2(h)
            k = F.relu(self.channel_key(h)) ** 2
            v = self.channel_value(k)
            r = torch.sigmoid(self.channel_receptance(h))
            h = residual + r * v
            return h

    class RWKVLM(nn.Module):
        def __init__(self, cfg: TinyModelConfig):
            super().__init__()
            self.cfg = cfg
            self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = nn.ModuleList([RWKVBlock(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_layers)])
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.token(input_ids)
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x))

    return RWKVLM(config)


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


def build_model(config: TinyModelConfig):
    if config.arch == "rwkv":
        return build_rwkv_model(config)
    return build_tiny_decoder_lm(config)


def save_checkpoint(model, config: TinyModelConfig, path: Path, step: int) -> None:
    import torch  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.to_dict(), "step": step}, path)


def load_checkpoint(path: Path, config: TinyModelConfig):
    import torch  # type: ignore

    model = build_model(config)
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
