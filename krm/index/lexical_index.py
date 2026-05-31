from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from krm.concepts.concept_card import ConceptCard


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,}")
STOPWORDS = {
    "and",
    "are",
    "because",
    "from",
    "have",
    "into",
    "that",
    "the",
    "their",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


@dataclass
class LexicalIndex:
    postings: dict[str, dict[int, int]] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    aliases: dict[int, list[str]] = field(default_factory=dict)

    @classmethod
    def from_concepts(cls, concepts: list[ConceptCard]) -> "LexicalIndex":
        idx = cls()
        for card in concepts:
            idx.names[card.concept_id] = card.canonical_name
            idx.aliases[card.concept_id] = card.aliases
            terms = tokenize(card.canonical_name + " " + " ".join(card.aliases))
            for term in terms:
                idx.postings.setdefault(term, {})
                idx.postings[term][card.concept_id] = idx.postings[term].get(card.concept_id, 0) + 1
        return idx

    def search(self, query: str, limit: int = 100) -> dict[int, tuple[float, list[str]]]:
        q_terms = tokenize(query)
        scores: dict[int, float] = {}
        reasons: dict[int, list[str]] = {}
        doc_count = max(1, len(self.names))
        for term in q_terms:
            posting = self.postings.get(term, {})
            if not posting:
                continue
            idf = math.log(1 + doc_count / max(1, len(posting)))
            for cid, tf in posting.items():
                name = self.names.get(cid, "").lower()
                reason = "title match" if term in tokenize(name) else "alias/term match"
                scores[cid] = scores.get(cid, 0.0) + (1.0 + math.log(tf)) * idf
                reasons.setdefault(cid, []).append(reason)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return {cid: (score, sorted(set(reasons[cid]))) for cid, score in ranked}
