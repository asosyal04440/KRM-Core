from pathlib import Path
from typing import Iterator

from krm.source.source_pointer import SourceArticle
from krm.source.zim_backend import ZimArticleInfo, ZimBackendUnavailableError
from krm.source.zim_reader import ZimSourceReader


class FakeOpenedZim:
    def metadata(self) -> dict:
        return {"title": "Fake"}

    def iter_articles(self, limit: int | None = None) -> Iterator[ZimArticleInfo]:
        articles = [
            ZimArticleInfo("a1", "Coal Article", "A", "/coal", 100),
            ZimArticleInfo("a2", "Biology Article", "A", "/bio", 200),
        ]
        yield from articles[:limit]

    def get_article_text(self, article_id: str) -> str:
        return {
            "a1": "<p>Coal and steam engine transport support industry.</p>",
            "a2": "Photosynthesis produces glucose and oxygen.",
        }[article_id]


class FakeBackend:
    name = "fake"

    def is_available(self) -> bool:
        return True

    def explain_availability(self) -> str:
        return "fake backend available"

    def open(self, path: Path):
        return FakeOpenedZim()


class MissingBackend:
    name = "missing"

    def is_available(self) -> bool:
        return False

    def explain_availability(self) -> str:
        return "missing backend"

    def open(self, path: Path):
        raise ZimBackendUnavailableError("missing backend")


def test_zim_reader_backend_missing_controlled_exception(tmp_path: Path) -> None:
    reader = ZimSourceReader(tmp_path / "fake.zim", backend=MissingBackend())
    try:
        list(reader.iter_articles())
    except ZimBackendUnavailableError as exc:
        assert "missing backend" in str(exc)
    else:
        raise AssertionError("missing backend must fail clearly")


def test_zim_reader_emits_articles_caps_chars_and_pointer(tmp_path: Path) -> None:
    path = tmp_path / "fake.zim"
    path.write_bytes(b"fake")
    reader = ZimSourceReader(path, max_articles=1, max_article_chars=12, backend=FakeBackend())
    articles = list(reader.iter_articles())
    assert len(articles) == 1
    assert isinstance(articles[0], SourceArticle)
    assert articles[0].text == "Coal and ste"
    pointer = articles[0].pointer()
    assert pointer.source_type == "zim"
    assert pointer.extra["zim_path"].endswith("fake.zim")
    assert pointer.extra["zim_url"] == "/coal"
