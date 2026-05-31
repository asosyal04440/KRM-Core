from krm.training.model_config import TinyModelConfig
from krm.training.torch_backend import build_tiny_decoder_lm, torch_availability


def test_torch_backend_imports_without_required_torch() -> None:
    status = torch_availability()
    assert "available" in status
    assert "message" in status


def test_torch_backend_tiny_forward_if_available() -> None:
    status = torch_availability()
    if not status["available"]:
        assert "not installed" in status["message"]
        return
    import torch  # type: ignore

    cfg = TinyModelConfig("TEST", vocab_size=32, max_seq_len=8, n_layers=1, n_heads=1, d_model=16, d_ff=32, dropout=0.0)
    model = build_tiny_decoder_lm(cfg)
    out = model(torch.zeros((1, 4), dtype=torch.long))
    assert tuple(out.shape) == (1, 4, 32)
