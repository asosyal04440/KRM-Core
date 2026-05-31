from pathlib import Path

from krm.source.dataset_readers import DatasetReaderConfig, reader_for_path


def test_dataset_readers_stream_and_cap_jsonl_json_csv_tsv(tmp_path: Path) -> None:
    jsonl = tmp_path / "qa.jsonl"
    jsonl.write_text('{"question":"Q1","answer":"A1"}\n{"question":"Q2","answer":"A2"}\n', encoding="utf-8")
    assert len(list(reader_for_path(jsonl, DatasetReaderConfig(max_rows=1)).iter_rows())) == 1

    js = tmp_path / "data.json"
    js.write_text('{"rows":[{"text":"alpha"},{"text":"beta"}]}', encoding="utf-8")
    assert len(list(reader_for_path(js).iter_rows())) == 2

    csv = tmp_path / "data.csv"
    csv.write_text("title,text\nA,alpha\n", encoding="utf-8")
    assert list(reader_for_path(csv).iter_rows())[0].title == "A"

    tsv = tmp_path / "data.tsv"
    tsv.write_text("title\ttext\nB\tbeta\n", encoding="utf-8")
    assert list(reader_for_path(tsv).iter_rows())[0].text == "beta"


def test_dataset_reader_missing_fields_skips(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text('{"x":"y"}\n', encoding="utf-8")
    reader = reader_for_path(path)
    rows = list(reader.iter_rows())
    assert rows == []
    assert reader.warnings
