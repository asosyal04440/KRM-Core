from pathlib import Path

from krm.source.source_pointer import SourcePointer
from krm.source.zim_reader import PlainTextSourceReader, StubZimSourceReader


def test_source_pointer_serialization_and_plain_text_retrieval(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    doc = source / "coal.txt"
    doc.write_text("Coal powered early steam engines.", encoding="utf-8")
    article = next(PlainTextSourceReader(source).iter_articles())
    pointer = article.pointer()
    restored = SourcePointer.from_compact_dict(pointer.to_compact_dict())
    assert restored.pointer_id == pointer.pointer_id
    assert PlainTextSourceReader(source).get_text(restored) == "Coal powered early steam engines."


def test_zim_reader_is_explicit_stub(tmp_path: Path) -> None:
    reader = StubZimSourceReader(tmp_path)
    try:
        list(reader.iter_articles())
    except NotImplementedError as exc:
        assert "roadmap" in str(exc)
    else:
        raise AssertionError("StubZimSourceReader must not pretend ZIM parsing works")
