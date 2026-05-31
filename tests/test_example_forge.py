import json
from pathlib import Path

from krm.pipeline import build_shards
from krm.training.example_forge import forge_examples
from tests.test_ingest_dataset_cli import _fixtures
import subprocess
import sys


def test_example_forge_exports_jsonl_and_dry_run(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = _fixtures(tmp_path)
    mind = tmp_path / "mind_dataset"
    subprocess.run([sys.executable, "scripts/ingest_dataset.py", "--source", str(source), "--out", str(mind)], cwd=repo, check=True)
    build_shards(mind, "tiny")
    dry = forge_examples(mind, tmp_path / "training", max_examples=5, dry_run=True)
    assert dry["example_count"] == 5
    result = forge_examples(mind, tmp_path / "training", max_examples=5, types={"router", "domain", "eval"})
    path = tmp_path / "training" / "examples.jsonl"
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == result["example_count"]
    assert {"example_id", "type", "input", "target", "source_refs", "confidence", "notes"}.issubset(rows[0])
