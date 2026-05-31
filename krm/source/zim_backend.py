from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol


class ZimBackendError(RuntimeError):
    pass


class ZimBackendUnavailableError(ZimBackendError):
    pass


@dataclass(slots=True)
class ZimArticleInfo:
    article_id: str
    title: str
    namespace: str | None = None
    url: str | None = None
    size_hint: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenedZim(Protocol):
    def metadata(self) -> dict[str, Any]:
        ...

    def iter_articles(self, limit: int | None = None) -> Iterator[ZimArticleInfo]:
        ...

    def get_article_text(self, article_id: str) -> str:
        ...


class ZimBackend(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def explain_availability(self) -> str:
        ...

    def open(self, path: Path) -> OpenedZim:
        ...


class StubZimBackend:
    name = "stub"

    def is_available(self) -> bool:
        return False

    def explain_availability(self) -> str:
        return (
            "ZIM file detected, but no real ZIM parsing backend is installed. "
            "V0.3 backend is optional. Install/enable a supported backend later."
        )

    def open(self, path: Path) -> OpenedZim:
        raise ZimBackendUnavailableError(self.explain_availability())


class AutoZimBackend:
    name = "auto"

    def __init__(self) -> None:
        self._backend: ZimBackend = self._select_backend()

    def is_available(self) -> bool:
        return self._backend.is_available()

    def explain_availability(self) -> str:
        return self._backend.explain_availability()

    def open(self, path: Path) -> OpenedZim:
        return self._backend.open(path)

    @property
    def selected_backend_name(self) -> str:
        return self._backend.name

    def _select_backend(self) -> ZimBackend:
        if _find_spec("libzim.reader") is not None:
            return LibzimReaderBackend()
        if _find_spec("pyzim.archive") is not None:
            return PyzimBackend()
        return StubZimBackend()


class LibzimReaderBackend:
    name = "libzim.reader"

    def is_available(self) -> bool:
        return _find_spec("libzim.reader") is not None

    def explain_availability(self) -> str:
        return "Using optional libzim.reader backend." if self.is_available() else StubZimBackend().explain_availability()

    def open(self, path: Path) -> OpenedZim:
        if not self.is_available():
            raise ZimBackendUnavailableError(self.explain_availability())
        try:
            from libzim.reader import Archive  # type: ignore

            return _GenericOpenedZim(Archive(str(path)), path)
        except Exception as exc:  # pragma: no cover - depends on optional backend
            raise ZimBackendError(f"Failed to open ZIM with libzim.reader: {exc}") from exc


class PyzimBackend:
    name = "pyzim"

    def is_available(self) -> bool:
        return _find_spec("pyzim.archive") is not None

    def explain_availability(self) -> str:
        return "Using optional pyzim backend." if self.is_available() else StubZimBackend().explain_availability()

    def open(self, path: Path) -> OpenedZim:
        if not self.is_available():
            raise ZimBackendUnavailableError(self.explain_availability())
        try:
            from pyzim.archive import Archive  # type: ignore

            return _GenericOpenedZim(Archive.open(str(path)), path)
        except Exception as exc:  # pragma: no cover - depends on optional backend
            raise ZimBackendError(f"Failed to open ZIM with pyzim: {exc}") from exc


class _GenericOpenedZim:
    """Best-effort adapter for optional local ZIM libraries."""

    def __init__(self, archive: Any, path: Path) -> None:
        self.archive = archive
        self.path = path

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"path": str(self.path)}
        for key in ["Title", "Description", "Language", "Creator", "Publisher", "Date", "Name"]:
            value = self._try_metadata(key)
            if value:
                metadata[key.lower()] = value
        return metadata

    def iter_articles(self, limit: int | None = None) -> Iterator[ZimArticleInfo]:
        count = 0
        for entry in self._iter_entries():
            title = str(getattr(entry, "title", "") or getattr(entry, "path", "") or getattr(entry, "url", "") or "")
            if not title:
                continue
            namespace = getattr(entry, "namespace", None)
            url = str(getattr(entry, "url", "") or getattr(entry, "path", "") or "")
            article_id = str(getattr(entry, "entry_id", "") or getattr(entry, "index", "") or url or title)
            size_hint = getattr(entry, "size", None)
            yield ZimArticleInfo(article_id=article_id, title=title, namespace=str(namespace) if namespace is not None else None, url=url or None, size_hint=size_hint)
            count += 1
            if limit is not None and count >= limit:
                return

    def get_article_text(self, article_id: str) -> str:
        entry = self._get_entry(article_id)
        if entry is None:
            return ""
        content = self._entry_content(entry)
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content or "")

    def _try_metadata(self, key: str) -> str | None:
        for method_name in ["get_metadata", "metadata"]:
            method = getattr(self.archive, method_name, None)
            if callable(method):
                try:
                    value = method(key)
                    if isinstance(value, bytes):
                        return value.decode("utf-8", errors="replace")
                    if value:
                        return str(value)
                except Exception:
                    pass
        return None

    def _iter_entries(self) -> Iterator[Any]:
        for method_name in ["iter_entries", "iter_articles", "entries"]:
            method = getattr(self.archive, method_name, None)
            if callable(method):
                try:
                    yield from method()
                    return
                except Exception:
                    continue
        for attr_name in ["entries", "articles"]:
            value = getattr(self.archive, attr_name, None)
            if value is not None:
                try:
                    yield from value
                    return
                except Exception:
                    continue
        return

    def _get_entry(self, article_id: str) -> Any | None:
        for method_name in ["get_entry_by_url", "get_entry_by_path", "get_article_by_url", "get_entry"]:
            method = getattr(self.archive, method_name, None)
            if callable(method):
                try:
                    return method(article_id)
                except Exception:
                    continue
        for entry in self._iter_entries():
            if article_id in {str(getattr(entry, "entry_id", "")), str(getattr(entry, "url", "")), str(getattr(entry, "path", "")), str(getattr(entry, "title", ""))}:
                return entry
        return None

    def _entry_content(self, entry: Any) -> Any:
        for method_name in ["read", "get_content", "content"]:
            method = getattr(entry, method_name, None)
            if callable(method):
                return method()
        return getattr(entry, "content", "")


def backend_from_name(name: str) -> ZimBackend:
    normalized = name.lower()
    if normalized == "auto":
        return AutoZimBackend()
    if normalized in {"stub", "none"}:
        return StubZimBackend()
    if normalized in {"libzim", "libzim.reader"}:
        return LibzimReaderBackend()
    if normalized == "pyzim":
        return PyzimBackend()
    raise ValueError(f"unsupported ZIM backend: {name}")


def _find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
