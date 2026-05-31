from __future__ import annotations


class LocalModel:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Real local GGUF/ONNX model support is a roadmap item for KRM-Core V0.")
