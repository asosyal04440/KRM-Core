from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from krm.source.dataset_discovery import SKIP_DIRS


INGESTIBLE_DATASET_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".md", ".markdown", ".html", ".htm"}
DATASET_LIKE_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv"}
DETECTED_UNSUPPORTED_SUFFIXES = {".parquet", ".arrow", ".sqlite", ".db", ".zip", ".gz"}


@dataclass(slots=True)
class DatasetFile:
    path: Path
    suffix: str
    size_bytes: int
    supported: bool
    ingestible: bool
    dataset_like: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["size_mb"] = round(self.size_bytes / (1024 * 1024), 3)
        return data


class DatasetBridge:
    def scan(self, folder: Path, recursive: bool = True) -> list[DatasetFile]:
        root = Path(folder)
        if not root.exists():
            return [DatasetFile(root, "", 0, False, False, False, "folder does not exist; create it and place local dataset files there")]
        if not root.is_dir():
            return [DatasetFile(root, root.suffix.lower(), root.stat().st_size, False, False, False, "source is not a folder")]
        pattern = "**/*" if recursive else "*"
        files: list[DatasetFile] = []
        for path in sorted(root.glob(pattern), key=lambda p: str(p).lower()):
            if not path.is_file() or self._skip(path, root):
                continue
            suffix = path.suffix.lower()
            size = path.stat().st_size
            if suffix in INGESTIBLE_DATASET_SUFFIXES:
                files.append(DatasetFile(path.resolve(), suffix, size, True, True, suffix in DATASET_LIKE_SUFFIXES, "ingestible local dataset/text format"))
            elif suffix in DETECTED_UNSUPPORTED_SUFFIXES:
                files.append(DatasetFile(path.resolve(), suffix, size, True, False, True, "detected but not supported in V0.4"))
            else:
                files.append(DatasetFile(path.resolve(), suffix, size, False, False, False, "unsupported suffix"))
        return files

    def _skip(self, path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        return any(part.startswith(".") or part in SKIP_DIRS for part in relative.parts[:-1])


def summarize_dataset_files(files: list[DatasetFile]) -> dict[str, Any]:
    real = [item for item in files if item.path.suffix]
    ingestible = [item for item in real if item.ingestible]
    unsupported = [item for item in real if not item.ingestible]
    return {
        "total_files": len(real),
        "ingestible_files": len(ingestible),
        "unsupported_files": len(unsupported),
        "dataset_like_files": len([item for item in real if item.dataset_like]),
        "total_size_bytes": sum(item.size_bytes for item in real),
        "ingestible_size_bytes": sum(item.size_bytes for item in ingestible),
        "unsupported_size_bytes": sum(item.size_bytes for item in unsupported),
        "warnings": sorted({item.reason for item in unsupported if item.supported}),
    }
