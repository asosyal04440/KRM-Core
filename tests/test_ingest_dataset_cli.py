import json
import subprocess
import sys
from pathlib import Path


def _fixtures(tmp_path: Path) -> Path:
    source = tmp_path / "datasets"
    source.mkdir()
    (source / "qa.jsonl").write_text('{"question":"Q?","answer":"A","title":"T"}\n', encoding="utf-8")
    (source / "data.json").write_text('{"rows":[{"instruction":"Do","input":"X","output":"Y"}]}', encoding="utf-8")
    (source / "rows.csv").write_text("title,text\nA,alpha body\n", encoding="utf-8")
    (source / "rows.tsv").write_text("title\ttext\nB\tbeta body\n", encoding="utf-8")
    (source / "big.jsonl").write_text("x" * 2048, encoding="utf-8")
    return source


def test_ingest_dataset_dry_run_and_ingest(tmp_path: Path) -> None:
    source = _fixtures(tmp_path)
    out = tmp_path / "mind_dataset"
    repo = Path(__file__).resolve().parents[1]
    dry = subprocess.run([sys.executable, "scripts/ingest_dataset.py", "--source", str(source), "--out", str(out), "--dry-run", "--json"], cwd=repo, text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["dry_run"] is True
    assert not out.exists()
    subprocess.run([sys.executable, "scripts/ingest_dataset.py", "--source", str(source), "--out", str(out), "--max-file-mb", "1"], cwd=repo, text=True, capture_output=True, check=True)
    assert (out / "mind.skel" / "concepts.jsonl").exists()


def test_ingest_dataset_field_overrides(tmp_path: Path) -> None:
    source = tmp_path / "datasets"
    source.mkdir()
    (source / "custom.csv").write_text("name,body\nCustom,custom body\n", encoding="utf-8")
    out = tmp_path / "mind_dataset"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/ingest_dataset.py", "--source", str(source), "--out", str(out), "--title-field", "name", "--text-field", "body"], cwd=repo, text=True, capture_output=True, check=True)
    assert (out / "mind.skel" / "sources.jsonl").exists()
