import json
import subprocess
import sys
from pathlib import Path


def test_inspect_zim_no_files_and_json(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/inspect_zim.py", "--source", str(tmp_path), "--json"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["zim_file_count"] == 0


def test_inspect_zim_fake_file_reports_backend_status(tmp_path: Path) -> None:
    (tmp_path / "fake.zim").write_bytes(b"not a real zim")
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/inspect_zim.py", "--source", str(tmp_path)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "detected ZIM files: 1" in result.stdout
    assert "backend" in result.stdout
    assert "recommendation" in result.stdout
