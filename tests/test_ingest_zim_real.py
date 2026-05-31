import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator

from krm.pipeline import ingest_articles
from krm.source.zim_backend import ZimArticleInfo
from krm.source.zim_reader import ZimSourceReader


class FakeOpenedZim:
    def metadata(self) -> dict:
        return {"title": "Fake"}

    def iter_articles(self, limit: int | None = None) -> Iterator[ZimArticleInfo]:
        articles = [ZimArticleInfo("a1", "Coal Article", "A", "/coal", 100), ZimArticleInfo("a2", "Biology Article", "A", "/bio", 200)]
        yield from articles[:limit]

    def get_article_text(self, article_id: str) -> str:
        return {"a1": "Coal and steam engine transport support industry.", "a2": "Photosynthesis produces glucose and oxygen."}[article_id]


class FakeBackend:
    name = "fake"

    def is_available(self) -> bool:
        return True

    def explain_availability(self) -> str:
        return "fake backend available"

    def open(self, path: Path):
        return FakeOpenedZim()


def test_ingest_zim_real_dry_run_writes_nothing_without_backend(tmp_path: Path) -> None:
    source = tmp_path / "zims"
    source.mkdir()
    (source / "fake.zim").write_bytes(b"fake")
    out = tmp_path / "mind_zim"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/ingest_zim_real.py", "--source", str(source), "--out", str(out), "--dry-run", "--json", "--backend", "stub"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["backend_available"] is False
    assert not out.exists()
    assert any("no real ZIM parsing backend" in warning for warning in payload["warnings"])


def test_mock_backend_zim_ingestion_produces_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "fake.zim"
    path.write_bytes(b"fake")
    articles = list(ZimSourceReader(path, max_articles=2, backend=FakeBackend()).iter_articles())
    out = tmp_path / "mind_zim"
    stats = ingest_articles(articles, out)
    assert stats["articles"] == 2
    assert stats["concepts"] > 0
    assert (out / "mind.skel" / "concepts.jsonl").exists()
    assert not (out / "raw_text").exists()
