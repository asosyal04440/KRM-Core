from __future__ import annotations

import re
from dataclasses import dataclass


DOMAIN_IDS = {
    "general": 0,
    "history": 1,
    "biology": 2,
    "technology": 3,
    "climate": 4,
    "programming": 5,
    "chemistry": 6,
    "economics": 7,
    "geography": 8,
    "politics": 9,
    "education": 10,
}

DOMAIN_NAMES = {value: key for key, value in DOMAIN_IDS.items()}

DOMAIN_KEYWORDS = {
    "history": {
        "agriculture",
        "britain",
        "empire",
        "industrial",
        "labor",
        "ottoman",
        "printing press",
        "revolution",
        "trade",
    },
    "biology": {
        "cell",
        "cellular",
        "chloroplast",
        "glucose",
        "mitochondria",
        "oxygen",
        "photosynthesis",
        "respiration",
    },
    "chemistry": {"carbon dioxide", "chemical", "glucose", "oxygen", "water"},
    "technology": {"engine", "industry", "machine", "printing press", "steam", "technology", "transport"},
    "economics": {"capital", "labor", "market", "production", "trade"},
    "geography": {"britain", "climate", "coal", "ports", "rainfall", "soil", "transport"},
    "politics": {"bureaucracy", "censorship", "empire", "institutions", "political", "reform"},
    "education": {"book", "education", "literacy", "manual", "printing", "scientific"},
    "programming": {"code", "data", "function", "program", "programming", "variable"},
    "climate": {"climate", "crop", "harvest", "rainfall", "seasonal", "temperature"},
}


@dataclass(frozen=True, slots=True)
class DomainScore:
    domain_id: int
    name: str
    score: int
    matched_keywords: list[str]


def detect_domains(text: str, max_domains: int = 3) -> list[DomainScore]:
    normalized = " ".join(re.findall(r"[a-z0-9]+(?:\s+[a-z0-9]+)?", text.lower()))
    scores: list[DomainScore] = []
    for name, keywords in DOMAIN_KEYWORDS.items():
        matched = sorted(keyword for keyword in keywords if keyword in normalized)
        if matched:
            scores.append(DomainScore(DOMAIN_IDS[name], name, len(matched), matched))
    scores.sort(key=lambda item: (-item.score, item.name))
    return scores[:max_domains] or [DomainScore(DOMAIN_IDS["general"], "general", 0, [])]


def primary_domain_id(text: str) -> int:
    return detect_domains(text, max_domains=1)[0].domain_id


def domain_name(domain_id: int) -> str:
    return DOMAIN_NAMES.get(domain_id, "general")
