from __future__ import annotations

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

from krm.source.dataset_discovery import DatasetDiscovery, DiscoveredFile
from krm.source.source_pointer import SourceArticle, SourcePointer
from krm.source.text_normalizer import title_from_markdown
from krm.source.zim_backend import ZimBackend, ZimBackendError, ZimBackendUnavailableError, backend_from_name


class _FolderReader:
    suffixes: tuple[str, ...] = ()
    source_type = "plain_text"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def iter_articles(self) -> Iterator[SourceArticle]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            title = self._title(path, text)
            yield SourceArticle(
                source_id=self.root.name,
                source_type=self.source_type,
                article_id=path.stem,
                title=title,
                text=text,
                path=path.resolve(),
            )

    def get_text(self, pointer: SourcePointer) -> str:
        raw_path = pointer.extra.get("path")
        if not raw_path:
            return ""
        text = Path(raw_path).read_text(encoding="utf-8")
        start = pointer.char_start or 0
        end = pointer.char_end if pointer.char_end is not None else len(text)
        return text[start:end]

    def _title(self, path: Path, text: str) -> str:
        return path.stem.replace("_", " ").title()


class PlainTextSourceReader(_FolderReader):
    suffixes = (".txt",)
    source_type = "plain_text"


class MarkdownSourceReader(_FolderReader):
    suffixes = (".md", ".markdown")
    source_type = "markdown"

    def _title(self, path: Path, text: str) -> str:
        return title_from_markdown(text, path.stem.replace("_", " ").title())


class StubZimSourceReader:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def iter_articles(self) -> Iterator[SourceArticle]:
        raise NotImplementedError("Real Kiwix/ZIM parsing is a roadmap item for KRM-Core V0.")

    def get_text(self, pointer: SourcePointer) -> str:
        raise NotImplementedError("Real Kiwix/ZIM parsing is a roadmap item for KRM-Core V0.")


class ZimSourceReader:
    def __init__(
        self,
        path: Path | str,
        max_articles: int = 500,
        max_article_chars: int = 20_000,
        include_namespaces: list[str] | None = None,
        skip_empty: bool = True,
        backend: ZimBackend | None = None,
        backend_name: str = "auto",
        include_title_contains: str | None = None,
        exclude_title_contains: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.max_articles = max_articles
        self.max_article_chars = max_article_chars
        self.include_namespaces = include_namespaces
        self.skip_empty = skip_empty
        self.backend = backend or backend_from_name(backend_name)
        self.include_title_contains = include_title_contains.lower() if include_title_contains else None
        self.exclude_title_contains = exclude_title_contains.lower() if exclude_title_contains else None
        self.warnings: list[str] = []

    def iter_articles(self) -> Iterator[SourceArticle]:
        if not self.backend.is_available():
            raise ZimBackendUnavailableError(self.backend.explain_availability())
        try:
            opened = self.backend.open(self.path)
        except ZimBackendError:
            raise
        except Exception as exc:
            raise ZimBackendError(f"failed to open ZIM file: {exc}") from exc
        yielded = 0
        for info in opened.iter_articles(limit=self.max_articles):
            if self.include_namespaces is not None and info.namespace not in self.include_namespaces:
                continue
            title = info.title.strip()
            if not title:
                self.warnings.append(f"{info.article_id}: skipped because title is empty")
                continue
            title_lower = title.lower()
            if self.include_title_contains and self.include_title_contains not in title_lower:
                continue
            if self.exclude_title_contains and self.exclude_title_contains in title_lower:
                continue
            try:
                text = opened.get_article_text(info.article_id)
            except Exception as exc:
                self.warnings.append(f"{title}: skipped because text extraction failed: {exc}")
                continue
            text = strip_html(text) if "<" in text and ">" in text else re.sub(r"\s+", " ", text).strip()
            if self.skip_empty and not text:
                self.warnings.append(f"{title}: skipped because article text is empty")
                continue
            text = text[: self.max_article_chars]
            yielded += 1
            yield SourceArticle(
                source_id=self.path.stem,
                source_type="zim",
                article_id=info.article_id,
                title=title,
                text=text,
                path=self.path.resolve(),
                metadata={"zim_path": str(self.path.resolve()), "zim_url": info.url, "namespace": info.namespace},
            )
            if yielded >= self.max_articles:
                return

    def get_text(self, pointer: SourcePointer) -> str:
        if not self.backend.is_available():
            raise ZimBackendUnavailableError(self.backend.explain_availability())
        opened = self.backend.open(self.path)
        text = opened.get_article_text(pointer.article_id or "")
        if "<" in text and ">" in text:
            text = strip_html(text)
        start = pointer.char_start or 0
        end = pointer.char_end if pointer.char_end is not None else len(text)
        return text[start:end]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def strip_html(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = parser.text()
    if text:
        return text
    return re.sub(r"<[^>]+>", " ", html)


class HtmlSourceReader(_FolderReader):
    suffixes = (".html", ".htm")
    source_type = "html"

    def iter_articles(self) -> Iterator[SourceArticle]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = strip_html(raw)
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else self._title(path, text)
            yield SourceArticle(self.root.name, self.source_type, path.stem, title, text, path.resolve())


class JsonlSourceReader:
    def __init__(self, root: Path | str, title_field: str = "title", text_field: str = "text") -> None:
        self.root = Path(root)
        self.title_field = title_field
        self.text_field = text_field
        self.warnings: list[str] = []

    def iter_articles(self) -> Iterator[SourceArticle]:
        files = [self.root] if self.root.is_file() else sorted(self.root.rglob("*.jsonl"))
        for path in files:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for row_number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        self.warnings.append(f"{path}: line {row_number} is not valid JSON")
                        continue
                    title = obj.get(self.title_field)
                    text = obj.get(self.text_field)
                    if not title or not text:
                        self.warnings.append(f"{path}: line {row_number} missing {self.title_field}/{self.text_field}")
                        continue
                    yield SourceArticle(
                        source_id=path.parent.name,
                        source_type="jsonl",
                        article_id=f"{path.stem}-{row_number}",
                        title=str(title),
                        text=str(text),
                        path=path.resolve(),
                        metadata={"row": row_number, "title_field": self.title_field, "text_field": self.text_field},
                    )

    def get_text(self, pointer: SourcePointer) -> str:
        raw_path = pointer.extra.get("path")
        row = pointer.extra.get("row")
        if not raw_path or row is None:
            return ""
        for article in self.__class__(Path(raw_path), pointer.extra.get("title_field", "title"), pointer.extra.get("text_field", "text")).iter_articles():
            if article.metadata.get("row") == row:
                return article.text
        return ""


class CsvSourceReader:
    def __init__(self, root: Path | str, title_field: str = "title", text_field: str = "text") -> None:
        self.root = Path(root)
        self.title_field = title_field
        self.text_field = text_field
        self.warnings: list[str] = []

    def iter_articles(self) -> Iterator[SourceArticle]:
        files = [self.root] if self.root.is_file() else sorted(self.root.rglob("*.csv"))
        for path in files:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or self.title_field not in reader.fieldnames or self.text_field not in reader.fieldnames:
                    self.warnings.append(f"{path}: missing required columns {self.title_field}/{self.text_field}")
                    continue
                for row_number, row in enumerate(reader, start=2):
                    title = row.get(self.title_field)
                    text = row.get(self.text_field)
                    if not title or not text:
                        self.warnings.append(f"{path}: row {row_number} missing {self.title_field}/{self.text_field}")
                        continue
                    yield SourceArticle(
                        source_id=path.parent.name,
                        source_type="csv",
                        article_id=f"{path.stem}-{row_number}",
                        title=str(title),
                        text=str(text),
                        path=path.resolve(),
                        metadata={"row": row_number, "title_field": self.title_field, "text_field": self.text_field},
                    )

    def get_text(self, pointer: SourcePointer) -> str:
        raw_path = pointer.extra.get("path")
        row = pointer.extra.get("row")
        if not raw_path or row is None:
            return ""
        for article in self.__class__(Path(raw_path), pointer.extra.get("title_field", "title"), pointer.extra.get("text_field", "text")).iter_articles():
            if article.metadata.get("row") == row:
                return article.text
        return ""


class LocalFolderSourceReader:
    def __init__(
        self,
        root: Path | str,
        recursive: bool = True,
        max_files: int = 100,
        max_articles: int = 10_000,
        max_file_mb: float = 25.0,
        include_suffixes: set[str] | None = None,
        jsonl_title_field: str = "title",
        jsonl_text_field: str = "text",
        csv_title_field: str = "title",
        csv_text_field: str = "text",
    ) -> None:
        self.root = Path(root)
        self.recursive = recursive
        self.max_files = max_files
        self.max_articles = max_articles
        self.max_file_bytes = int(max_file_mb * 1024 * 1024)
        self.include_suffixes = {suffix.lower() for suffix in include_suffixes} if include_suffixes else None
        self.jsonl_title_field = jsonl_title_field
        self.jsonl_text_field = jsonl_text_field
        self.csv_title_field = csv_title_field
        self.csv_text_field = csv_text_field
        self.warnings: list[str] = []

    def discover(self) -> list[DiscoveredFile]:
        files = DatasetDiscovery().scan(self.root, recursive=self.recursive)
        ingestible = []
        for item in files:
            if self.include_suffixes is not None and item.suffix not in self.include_suffixes:
                continue
            if item.suffix == ".zim":
                self.warnings.append(item.reason)
                continue
            if not item.ingestible:
                continue
            if item.size_bytes > self.max_file_bytes:
                self.warnings.append(f"{item.path}: skipped because file exceeds max size")
                continue
            ingestible.append(item)
            if len(ingestible) >= self.max_files:
                break
        return ingestible

    def iter_articles(self) -> Iterator[SourceArticle]:
        count = 0
        for item in self.discover():
            for article in self._iter_file(item.path, item.suffix):
                if count >= self.max_articles:
                    self.warnings.append(f"stopped at max articles cap: {self.max_articles}")
                    return
                count += 1
                yield article

    def _iter_file(self, path: Path, suffix: str) -> Iterator[SourceArticle]:
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            yield SourceArticle(path.parent.name, "plain_text", path.stem, path.stem.replace("_", " ").title(), text, path.resolve())
        elif suffix in {".md", ".markdown"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield SourceArticle(path.parent.name, "markdown", path.stem, title_from_markdown(text, path.stem.replace("_", " ").title()), text, path.resolve())
        elif suffix in {".html", ".htm"}:
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = strip_html(raw)
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else path.stem.replace("_", " ").title()
            yield SourceArticle(path.parent.name, "html", path.stem, title, text, path.resolve())
        elif suffix == ".jsonl":
            reader = JsonlSourceReader(path, self.jsonl_title_field, self.jsonl_text_field)
            yield from reader.iter_articles()
            self.warnings.extend(reader.warnings)
        elif suffix == ".csv":
            reader = CsvSourceReader(path, self.csv_title_field, self.csv_text_field)
            yield from reader.iter_articles()
            self.warnings.extend(reader.warnings)

    def get_text(self, pointer: SourcePointer) -> str:
        raw_path = pointer.extra.get("path")
        if not raw_path:
            return ""
        path = Path(raw_path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return JsonlSourceReader(path, pointer.extra.get("title_field", "title"), pointer.extra.get("text_field", "text")).get_text(pointer)
        if suffix == ".csv":
            return CsvSourceReader(path, pointer.extra.get("title_field", "title"), pointer.extra.get("text_field", "text")).get_text(pointer)
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix in {".html", ".htm"}:
            text = strip_html(text)
        start = pointer.char_start or 0
        end = pointer.char_end if pointer.char_end is not None else len(text)
        return text[start:end]
