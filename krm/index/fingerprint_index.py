from __future__ import annotations

import hashlib

from krm.index.lexical_index import tokenize


def fingerprint_text(text: str) -> int:
    bits = [0] * 64
    for term in tokenize(text):
        digest = int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            bits[i] += 1 if digest & (1 << i) else -1
    fp = 0
    for i, value in enumerate(bits):
        if value >= 0:
            fp |= 1 << i
    return fp


def fingerprint_similarity(left: int, right: int) -> float:
    distance = (left ^ right).bit_count()
    return 1.0 - (distance / 64.0)
