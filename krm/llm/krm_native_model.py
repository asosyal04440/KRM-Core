from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class KRMNativeModel:
    def __init__(self, export_dir: Path) -> None:
        self.export_dir = export_dir

    def metadata(self) -> dict[str, Any]:
        manifest = self.export_dir / "export_manifest.json"
        if not manifest.exists():
            return {"available": False, "message": f"KRM-native export metadata not found: {manifest}"}
        return {"available": True, "metadata": json.loads(manifest.read_text(encoding="utf-8"))}

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("KRM-native inference integration is future work unless the optional torch backend is wired in.")
