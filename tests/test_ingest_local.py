import json
import subprocess
import sys
from pathlib import Path

from krm.pipeline import build_shards, run_query


def _local_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "local"
    source.mkdir()
    (source / "a.txt").write_text("Coal and steam engine transport support textile industry.", encoding="utf-8")
    (source / "b.html").write_text("<title>Biology</title><p>Photosynthesis produces glucose and oxygen.</p>", encoding="utf-8")
    (source / "c.jsonl").write_text('{"title":"Ottoman Printing","text":"Printing press could support literacy and bureaucracy."}\n', encoding="utf-8")
    (source / "d.csv").write_text("title,text\nCell Energy,Cellular respiration uses glucose and oxygen.\n", encoding="utf-8")
    (source / "e.zim").write_text("not parsed", encoding="utf-8")
    return source


def test_ingest_local_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = _local_fixture(tmp_path)
    out = tmp_path / "mind"
    result = subprocess.run(
        [sys.executable, "scripts/ingest_local.py", "--source", str(source), "--out", str(out), "--dry-run", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert not out.exists()
    assert any("ZIM detected" in warning for warning in payload["warnings"])


def test_ingest_local_caps_and_outputs_concepts(tmp_path: Path) -> None:
    source = _local_fixture(tmp_path)
    out = tmp_path / "mind"
    subprocess.run(
        [sys.executable, "scripts/ingest_local.py", "--source", str(source), "--out", str(out), "--max-files", "4"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert (out / "mind.skel" / "concepts.jsonl").exists()
    build_shards(out, "tiny")
    result = run_query(out, "steam engine and coal", "tiny")
    assert "final_answer" in result
