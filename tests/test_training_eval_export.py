import json

from krm.training.eval_tiny_core import evaluate_tiny_core
from krm.training.export import export_tiny_core
from krm.training.model_config import get_model_config


def test_eval_fails_gracefully_without_model(tmp_path) -> None:
    report = evaluate_tiny_core(tmp_path / "missing", tmp_path / "tok", tmp_path / "corpus")
    assert report["ok"] is False
    assert "not found" in report["error"]


def test_export_copies_metadata_when_artifacts_exist(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    get_model_config("10m").save(model / "config.json")
    (model / "run_metadata.json").write_text(json.dumps({"step": 0}), encoding="utf-8")
    result = export_tiny_core(model, tmp_path / "export")
    assert result["ok"] is True
    assert (tmp_path / "export" / "export_manifest.json").exists()
    assert (tmp_path / "export" / "README_MODEL.md").exists()
