import json
import subprocess
import sys
from pathlib import Path


def test_inspect_data_human_and_json_output(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("local text", encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    human = subprocess.run(
        [sys.executable, "scripts/inspect_data.py", "--source", str(tmp_path)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "total files scanned" in human.stdout
    js = subprocess.run(
        [sys.executable, "scripts/inspect_data.py", "--source", str(tmp_path), "--json"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(js.stdout)
    assert payload["summary"]["ingestible_files"] == 1
