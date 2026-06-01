from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, AddedToken, trainers, pre_tokenizers, normalizers, decoders, processors
from tokenizers.models import BPE

from krm.training.special_tokens import list_special_tokens, validate_no_collision


PAD_TOKEN = "<|PAD|>"
UNK_TOKEN = "<|UNK|>"
BOS_TOKEN = "<|BOS|>"
EOS_TOKEN = "<|EOS|>"

DEFAULT_EXTRA_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class BPETokenizer:
    def __init__(self, vocab_size: int = 32768, special_tokens: list[str] | None = None) -> None:
        self._vocab_size = vocab_size
        if special_tokens is not None:
            self._special_tokens = special_tokens
            has_all_base = all(t in special_tokens for t in list_special_tokens())
            validate_no_collision(special_tokens, full_list=has_all_base)
        else:
            extra = DEFAULT_EXTRA_TOKENS
            validate_no_collision(extra)
            self._special_tokens = list_special_tokens() + extra
        self._tokenizer: Tokenizer | None = None

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return len(self._special_tokens)
        return int(self._tokenizer.get_vocab_size())

    def train(self, files: list[str], min_frequency: int = 2) -> None:
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        special = [AddedToken(t, single_word=False, normalized=False) for t in self._special_tokens]
        tokenizer.add_special_tokens(special)

        trainer = trainers.BpeTrainer(
            vocab_size=self._vocab_size,
            min_frequency=min_frequency,
            special_tokens=[str(t) for t in special],
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train(files, trainer)
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer not trained")
        output = self._tokenizer.encode(text)
        return output.ids

    def decode(self, ids: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer not trained")
        return self._tokenizer.decode(ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer not trained")
        return [output.ids for output in self._tokenizer.encode_batch(texts)]

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer not trained")
        return self._tokenizer.decode_batch(batch)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._tokenizer is not None:
            self._tokenizer.save(str(path / "tokenizer.json"))
        meta = {
            "type": "BPETokenizer",
            "vocab_size": self.vocab_size,
            "special_tokens": self._special_tokens,
            "trained": self._tokenizer is not None,
        }
        (path / "bpe_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        meta_path = path / "bpe_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            obj = cls(vocab_size=meta.get("vocab_size", 32768), special_tokens=meta.get("special_tokens"))
        else:
            obj = cls()
        tokenizer_file = path / "tokenizer.json"
        if tokenizer_file.exists():
            obj._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        return obj

    def token_to_id(self, token: str) -> int | None:
        if self._tokenizer is None:
            return None
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        if self._tokenizer is None:
            return None
        return self._tokenizer.id_to_token(token_id)

    def get_vocab(self) -> dict[str, int]:
        if self._tokenizer is None:
            return {}
        return self._tokenizer.get_vocab()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": self._special_tokens,
            "trained": self._tokenizer is not None,
        }
