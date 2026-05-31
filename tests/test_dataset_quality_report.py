from pathlib import Path

from krm.evals.dataset_quality import dataset_quality_report


def test_dataset_quality_report_duplicates_and_coverage(tmp_path: Path) -> None:
    source = tmp_path / "datasets"
    source.mkdir()
    (source / "rows.csv").write_text("title,text\nA,alpha\nA,alpha\nB,beta\n", encoding="utf-8")
    report = dataset_quality_report(source)
    item = report["files"][0]
    assert item["duplicate_row_estimate"] >= 1
    assert item["field_coverage"]["title"] == 3
