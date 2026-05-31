from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INGESTIBLE_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".jsonl", ".csv"}
SUPPORTED_SUFFIXES = INGESTIBLE_SUFFIXES | {".zim"}
SKIP_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__", "mind.cache", "mind.skel", "mind.index", "mind.shards", "mind.seeds"}


@dataclass(slots=True)
class DiscoveredFile:
    path: Path
    suffix: str
    size_bytes: int
    supported: bool
    ingestible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["size_mb"] = round(self.size_bytes / (1024 * 1024), 3)
        return data


class DatasetDiscovery:
    def scan(self, folder: Path, recursive: bool = True) -> list[DiscoveredFile]:
        root = Path(folder)
        if not root.exists():
            return [
                DiscoveredFile(
                    path=root,
                    suffix="",
                    size_bytes=0,
                    supported=False,
                    ingestible=False,
                    reason="folder does not exist",
                )
            ]
        if not root.is_dir():
            return [
                DiscoveredFile(
                    path=root,
                    suffix=root.suffix.lower(),
                    size_bytes=root.stat().st_size,
                    supported=False,
                    ingestible=False,
                    reason="source is not a folder",
                )
            ]
        pattern = "**/*" if recursive else "*"
        found: list[DiscoveredFile] = []
        for path in sorted(root.glob(pattern), key=lambda p: str(p).lower()):
            if not path.is_file() or self._is_skipped(path, root):
                continue
            suffix = path.suffix.lower()
            size = path.stat().st_size
            if suffix in INGESTIBLE_SUFFIXES:
                found.append(DiscoveredFile(path.resolve(), suffix, size, True, True, "ingestible local text format"))
            elif suffix == ".zim":
                found.append(
                    DiscoveredFile(
                        path.resolve(),
                        suffix,
                        size,
                        True,
                        False,
                        "ZIM detected but real ZIM parsing is not implemented in V0.2",
                    )
                )
            else:
                found.append(DiscoveredFile(path.resolve(), suffix, size, False, False, "unsupported suffix"))
        return found

    def _is_skipped(self, path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        return any(part.startswith(".") or part in SKIP_DIRS for part in relative.parts[:-1])


def summarize_discovery(files: list[DiscoveredFile]) -> dict[str, Any]:
    ingestible = [item for item in files if item.ingestible]
    non_ingestible = [item for item in files if not item.ingestible]
    total_size = sum(item.size_bytes for item in files if item.path.suffix)
    ingestible_size = sum(item.size_bytes for item in ingestible)
    skipped_size = sum(item.size_bytes for item in non_ingestible)
    warnings = [item.reason for item in files if item.suffix == ".zim"]
    return {
        "total_files_scanned": len([item for item in files if item.path.suffix]),
        "ingestible_files": len(ingestible),
        "skipped_files": len(non_ingestible),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 3),
        "total_ingestible_size_bytes": ingestible_size,
        "total_ingestible_size_mb": round(ingestible_size / (1024 * 1024), 3),
        "total_non_ingestible_size_bytes": skipped_size,
        "total_non_ingestible_size_mb": round(skipped_size / (1024 * 1024), 3),
        "estimated_artifact_size_bytes": max(4096, ingestible_size // 8) if ingestible else 0,
        "estimated_ram_impact_bytes": max(8 * 1024 * 1024, min(ingestible_size * 2, 512 * 1024 * 1024)) if ingestible else 0,
        "warnings": sorted(set(warnings)),
    }
