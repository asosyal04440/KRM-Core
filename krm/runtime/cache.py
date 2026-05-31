from __future__ import annotations

from pathlib import Path


class DisposableCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
