from pathlib import Path

from krm.source.dataset_discovery import DatasetDiscovery


def test_discovery_detects_supported_zim_and_unsupported(tmp_path: Path) -> None:
    for name in ["a.txt", "b.md", "c.html", "d.jsonl", "e.csv", "f.zim", "g.bin"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    files = DatasetDiscovery().scan(tmp_path)
    by_suffix = {item.suffix: item for item in files}
    assert by_suffix[".txt"].ingestible
    assert by_suffix[".md"].ingestible
    assert by_suffix[".html"].ingestible
    assert by_suffix[".jsonl"].ingestible
    assert by_suffix[".csv"].ingestible
    assert by_suffix[".zim"].supported is True
    assert by_suffix[".zim"].ingestible is False
    assert by_suffix[".bin"].supported is False


def test_discovery_handles_missing_folder(tmp_path: Path) -> None:
    result = DatasetDiscovery().scan(tmp_path / "missing")
    assert result[0].reason == "folder does not exist"
    assert result[0].ingestible is False
