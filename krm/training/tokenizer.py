from __future__ import annotations

import json
from pathlib import Path

from krm.training.special_tokens import list_special_tokens, validate_no_collision


class CharByteTokenizer:
    def __init__(self, special_tokens: list[str] | None = None) -> None:
        self.special_tokens = special_tokens or list_special_tokens()
        if special_tokens is None:
            validate_no_collision()
        elif len(self.special_tokens) != len(set(self.special_tokens)):
            raise ValueError("special token collision detected")
        self.token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.byte_offset = len(self.special_tokens)
        for value in range(256):
            self.token_to_id[f"<byte:{value}>"] = self.byte_offset + value
        self.id_to_token: dict[int, str] = {idx: token for token, idx in self.token_to_id.items()}
        self._special_by_length = sorted(self.special_tokens, key=len, reverse=True)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        index = 0
        raw = text
        while index < len(raw):
            matched = next((token for token in self._special_by_length if raw.startswith(token, index)), None)
            if matched is not None:
                ids.append(self.token_to_id[matched])
                index += len(matched)
                continue
            char = raw[index]
            for byte in char.encode("utf-8"):
                ids.append(self.byte_offset + byte)
            index += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        chunks: list[str] = []
        byte_buffer = bytearray()
        for token_id in ids:
            token = self.id_to_token.get(int(token_id))
            if token is None:
                continue
            if token.startswith("<byte:"):
                byte_buffer.append(int(token[len("<byte:") : -1]))
                continue
            if byte_buffer:
                chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()
            chunks.append(token)
        if byte_buffer:
            chunks.append(byte_buffer.decode("utf-8", errors="replace"))
        return "".join(chunks)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text(
            json.dumps({"type": "CharByteTokenizer", "special_tokens": self.special_tokens}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CharByteTokenizer":
        data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        return cls(list(data.get("special_tokens") or list_special_tokens()))


ByteLevelTokenizer = CharByteTokenizer
