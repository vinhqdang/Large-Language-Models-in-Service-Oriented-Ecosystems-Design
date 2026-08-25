import hashlib
from pathlib import Path

import pytest

from src.data.download import ChecksumMismatchError, download_file


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
