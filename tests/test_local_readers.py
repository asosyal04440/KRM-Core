from pathlib import Path

from krm.source.zim_reader import CsvSourceReader, HtmlSourceReader, JsonlSourceReader, LocalFolderSourceReader


def test_html_reader_strips_tags(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "page.html").write_text("<html><head><title>Hello</title></head><body><p>Coal <b>power</b></p></body></html>", encoding="utf-8")
    article = next(HtmlSourceReader(folder).iter_articles())
    assert article.title == "Hello"
    assert "Coal power" in article.text
    assert "<p>" not in article.text


def test_jsonl_reader_streams_rows_and_warns_on_missing(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"title":"A","text":"alpha text"}\n{"title":"B"}\n', encoding="utf-8")
    reader = JsonlSourceReader(path)
    rows = list(reader.iter_articles())
    assert len(rows) == 1
    assert rows[0].title == "A"
    assert reader.warnings


def test_csv_reader_uses_configured_fields(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text("name,body\nA,alpha body\n", encoding="utf-8")
    rows = list(CsvSourceReader(path, title_field="name", text_field="body").iter_articles())
    assert len(rows) == 1
    assert rows[0].title == "A"


def test_local_folder_reader_skips_zim(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Steam engine and coal.", encoding="utf-8")
    (tmp_path / "b.zim").write_text("not parsed", encoding="utf-8")
    reader = LocalFolderSourceReader(tmp_path)
    rows = list(reader.iter_articles())
    assert len(rows) == 1
    assert any("ZIM detected" in warning for warning in reader.warnings)
