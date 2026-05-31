from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol


@dataclass(slots=True)
class SourcePointer:
    source_id: str
    source_type: str
    title: str
    article_id: str | None = None
    section_id: str | None = None
    paragraph_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def pointer_id(self) -> str:
        parts = [
            self.source_id,
            self.article_id or "",
            self.section_id or "",
            self.paragraph_id or "",
            str(self.char_start if self.char_start is not None else ""),
            str(self.char_end if self.char_end is not None else ""),
        ]
        return "::".join(parts)

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "sid": self.source_id,
            "typ": self.source_type,
            "ttl": self.title,
            "aid": self.article_id,
            "sec": self.section_id,
            "par": self.paragraph_id,
            "cs": self.char_start,
            "ce": self.char_end,
            "x": self.extra,
        }

    @classmethod
    def from_compact_dict(cls, data: dict[str, Any]) -> "SourcePointer":
        return cls(
            source_id=data["sid"],
            source_type=data["typ"],
            title=data["ttl"],
            article_id=data.get("aid"),
            section_id=data.get("sec"),
            paragraph_id=data.get("par"),
            char_start=data.get("cs"),
            char_end=data.get("ce"),
            extra=dict(data.get("x") or {}),
        )


@dataclass(slots=True)
class SourceArticle:
    source_id: str
    source_type: str
    article_id: str
    title: str
    text: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def pointer(self) -> SourcePointer:
        extra: dict[str, Any] = dict(self.metadata)
        if self.path is not None:
            extra["path"] = str(self.path)
        return SourcePointer(
            source_id=self.source_id,
            source_type=self.source_type,
            title=self.title,
            article_id=self.article_id,
            char_start=0,
            char_end=len(self.text),
            extra=extra,
        )


class SourceReader(Protocol):
    def iter_articles(self) -> Iterator[SourceArticle]:
        ...

    def get_text(self, pointer: SourcePointer) -> str:
        ...
