from __future__ import annotations

import re
from dataclasses import dataclass, field

from krm.concepts.concept_card import ConceptCard
from krm.concepts.concept_types import ConceptType
from krm.concepts.domain import DOMAIN_IDS, primary_domain_id
from krm.index.fingerprint_index import fingerprint_text
from krm.index.lexical_index import STOPWORDS, tokenize
from krm.source.source_pointer import SourceArticle


KEY_PHRASES = [
    "Britain",
    "coal",
    "steam engine",
    "textile industry",
    "capital",
    "agriculture",
    "labor",
    "trade",
    "transport",
    "Industrial Revolution",
    "photosynthesis",
    "cellular respiration",
    "glucose",
    "oxygen",
    "carbon dioxide",
    "energy",
    "chloroplast",
    "mitochondria",
    "Ottoman Empire",
    "printing press",
    "literacy",
    "bureaucracy",
    "reform",
    "climate",
    "programming",
    "ATP",
    "water",
    "inputs",
    "outputs",
    "education",
    "resistance",
    "scientific exchange",
    "production",
    "markets",
    "ports",
    "canals",
]
REPEATED_TERM_ALLOW = {
    "agriculture",
    "bureaucracy",
    "capital",
    "climate",
    "energy",
    "glucose",
    "labor",
    "literacy",
    "oxygen",
    "programming",
    "reform",
    "trade",
    "transport",
}
CAPITALIZED_BLOCKLIST = {"Any", "Earlier", "More", "The", "This"}
TURKISH_STOPWORDS = {"bir", "ve", "ile", "bu", "su", "icin", "olan"}
GARBAGE_TERMS = STOPWORDS | TURKISH_STOPWORDS | {"usual", "common", "later", "mainly", "many", "some"}
TECHNICAL_PHRASE_RE = re.compile(
    r"\b(?:carbon dioxide|cellular respiration|steam engine|printing press|textile industry|industrial revolution|"
    r"source pointers?|ghost edges?|concept cards?)\b",
    flags=re.IGNORECASE,
)
CAUSE_PATTERNS = ["because", "led to", "caused", "enabled", "resulted in", "due to", "therefore", "allowed", "contributed to"]
COMPARISON_PATTERNS = ["compared with", "unlike", "whereas", "both", "difference", "similar", "linked"]


@dataclass(slots=True)
class EdgeCandidate:
    src_name: str
    dst_name: str
    edge_type: str
    reason: str


@dataclass(slots=True)
class ExtractionResult:
    concepts: list[ConceptCard]
    candidate_edges: list[EdgeCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


class ConceptExtractor:
    def extract(self, article: SourceArticle) -> ExtractionResult:
        text = article.text
        pointer_id = article.pointer().pointer_id
        domain_id = primary_domain_id(article.title + " " + article.text)
        scored_names = self._candidate_names(article)
        concepts = [
            ConceptCard(
                concept_id=-1,
                canonical_name=name,
                aliases=[],
                concept_type=self._type_for(name),
                domain_id=domain_id,
                importance=min(255, max(32, score)),
                confidence=210 if score >= 200 else 175,
                source_refs=[pointer_id],
                short_fingerprint=fingerprint_text(name + " " + text[:500]),
            )
            for name, score in scored_names
        ]
        names = [name for name, _ in scored_names]
        edges = self._candidate_edges(names, text)
        warnings = [] if concepts else ["no concepts extracted"]
        return ExtractionResult(
            concepts=concepts,
            candidate_edges=edges,
            warnings=warnings,
            stats={"concept_count": len(concepts), "edge_count": len(edges), "domain_id": domain_id},
        )

    def _candidate_names(self, article: SourceArticle) -> list[tuple[str, int]]:
        text = article.text
        found: dict[str, int] = {}
        self._add(found, article.title, 245)
        lower = text.lower()
        for phrase in KEY_PHRASES:
            if phrase.lower() in lower:
                self._add(found, phrase, 190)
        for match in TECHNICAL_PHRASE_RE.findall(text):
            self._add(found, match, 205)
        for heading in re.findall(r"^#+\s+(.+)$", text, flags=re.MULTILINE):
            self._add(found, heading, 230)
        for match in re.findall(r"\b[A-Z][A-Za-z]+(?:[- \t]+[A-Z][A-Za-z]+){0,3}\b", text):
            if len(match) > 3 and "\n" not in match and match not in CAPITALIZED_BLOCKLIST:
                self._add(found, match, 150)
        for match in re.findall(r"\b[a-z][a-z]+(?:-[a-z][a-z]+)+\b", text):
            self._add(found, match, 135)
        terms = tokenize(text)
        counts: dict[str, int] = {}
        for term in terms:
            if len(term) >= 5 and self._is_clean(term):
                counts[term] = counts.get(term, 0) + 1
        for term, count in counts.items():
            if count >= 2 and term in REPEATED_TERM_ALLOW:
                self._add(found, term, 130 + min(60, count * 10))
        ranked = sorted(found.items(), key=lambda item: (-item[1], item[0].lower()))
        return [(name, score) for name, score in ranked[:32]]

    def _candidate_edges(self, names: list[str], text: str) -> list[EdgeCandidate]:
        edges: list[EdgeCandidate] = []
        lower = text.lower()
        if any(pattern in lower for pattern in CAUSE_PATTERNS):
            edge_type = "CAUSES"
            reason = "cause-effect phrase pattern"
        elif any(pattern in lower for pattern in COMPARISON_PATTERNS):
            edge_type = "CONTRASTS"
            reason = "comparison phrase pattern"
        else:
            edge_type = "CO_OCCURS"
            reason = "same article co-occurrence"
        for i, src in enumerate(names[:16]):
            for dst in names[i + 1 : min(len(names), i + 6)]:
                edges.append(EdgeCandidate(src, dst, edge_type, reason))
        return edges

    def _type_for(self, name: str) -> ConceptType:
        lower = name.lower()
        if lower in {"photosynthesis", "cellular respiration", "programming"}:
            return ConceptType.PROCESS
        if lower in {"steam engine", "printing press"}:
            return ConceptType.TOOL
        if lower in {"britain", "ottoman empire"}:
            return ConceptType.PLACE
        if lower in {"industrial revolution"}:
            return ConceptType.EVENT
        return ConceptType.ENTITY

    def _add(self, found: dict[str, int], raw_name: str, score: int) -> None:
        name = self._normalize_name(raw_name)
        if not name:
            return
        key = name.lower()
        existing = next((old for old in found if old.lower() == key), None)
        if existing is None:
            found[name] = score
        else:
            found[existing] = max(found[existing], score)

    def _normalize_name(self, raw_name: str) -> str:
        name = re.sub(r"\s+", " ", raw_name.strip(" \t\r\n#.,;:!?()[]{}\"'"))
        if not self._is_clean(name):
            return ""
        phrase_map = {
            "atp": "ATP",
            "britain": "Britain",
            "ottoman empire": "Ottoman Empire",
            "industrial revolution": "Industrial Revolution",
            "cellular respiration": "cellular respiration",
            "photosynthesis": "photosynthesis",
        }
        return phrase_map.get(name.lower(), name)

    def _is_clean(self, name: str) -> bool:
        lower = name.lower().strip()
        if len(lower) < 3 or lower in GARBAGE_TERMS:
            return False
        if lower.isdigit():
            return False
        if not re.search(r"[a-zA-Z]", lower):
            return False
        parts = lower.replace("-", " ").split()
        if all(part in GARBAGE_TERMS for part in parts):
            return False
        return True
