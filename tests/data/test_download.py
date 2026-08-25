import hashlib

import pytest
import requests

from src.data.download import (
    ChecksumMismatchError,
    IncompleteDownloadError,
    download_file,
)


class _FakeResponse:
    def __init__(self, content: bytes):
        self._content = content
        self.headers = {"content-length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def test_download_writes_file_and_returns_path(tmp_path, monkeypatch):
    payload = b"hello adr corpus" * 1000
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest)

    assert result == dest
    assert dest.read_bytes() == payload


def test_download_verifies_correct_checksum(tmp_path, monkeypatch):
    payload = b"consistent bytes"
    correct_md5 = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest, expected_md5=correct_md5)

    assert result == dest


def test_download_raises_and_deletes_file_on_checksum_mismatch(tmp_path, monkeypatch):
    payload = b"tampered bytes"
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    with pytest.raises(ChecksumMismatchError):
        download_file("http://example.test/file", dest, expected_md5="0" * 32)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_retries_after_transient_failure_and_succeeds(tmp_path, monkeypatch):
    payload = b"retry succeeds" * 100
    calls = {"count": 0}

    def fake_get(url, stream, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ConnectionError("transient failure")
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data.download.requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest)

    assert result == dest
    assert dest.read_bytes() == payload
    assert calls["count"] == 2
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_raises_after_exhausting_retries_and_leaves_no_partial_file(
    tmp_path, monkeypatch
):
    def fake_get(url, stream, timeout):
        raise requests.exceptions.ConnectionError("persistent failure")

    monkeypatch.setattr("src.data.download.requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    dest = tmp_path / "out.bin"

    with pytest.raises(requests.exceptions.ConnectionError):
        download_file("http://example.test/file", dest)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_raises_incomplete_download_error_when_stream_is_truncated(
    tmp_path, monkeypatch
):
    class _TruncatedResponse(_FakeResponse):
        def __init__(self, content: bytes, declared_length: int):
            super().__init__(content)
            self.headers = {"content-length": str(declared_length)}

    payload = b"short content"
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout: _TruncatedResponse(
            payload, declared_length=len(payload) + 100
        ),
    )
    dest = tmp_path / "out.bin"

    with pytest.raises(IncompleteDownloadError):
        download_file("http://example.test/file", dest)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
