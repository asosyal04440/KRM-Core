from pathlib import Path

from krm.source.zim_backend import AutoZimBackend, StubZimBackend, ZimBackendUnavailableError


def test_stub_zim_backend_reports_unavailable() -> None:
    backend = StubZimBackend()
    assert backend.is_available() is False
    assert "no real ZIM parsing backend" in backend.explain_availability()
    try:
        backend.open(Path("missing.zim"))
    except ZimBackendUnavailableError as exc:
        assert "optional" in str(exc)
    else:
        raise AssertionError("stub backend must not open files")


def test_auto_zim_backend_does_not_crash_without_optional_backend() -> None:
    backend = AutoZimBackend()
    assert isinstance(backend.is_available(), bool)
    assert backend.explain_availability()
