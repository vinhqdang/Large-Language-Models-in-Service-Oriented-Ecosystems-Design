import hashlib

import pytest
import requests

from src.data.download import (
    ChecksumMismatchError,
    IncompleteDownloadError,
    download_file,
)


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self._content = content
        self.headers = {"content-length": str(len(content))}
        self.status_code = status_code

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
        lambda url, stream, timeout, headers=None: _FakeResponse(payload),
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
        lambda url, stream, timeout, headers=None: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest, expected_md5=correct_md5)

    assert result == dest


def test_download_raises_and_deletes_file_on_checksum_mismatch(tmp_path, monkeypatch):
    payload = b"tampered bytes"
    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout, headers=None: _FakeResponse(payload),
    )
    dest = tmp_path / "out.bin"

    with pytest.raises(ChecksumMismatchError):
        download_file("http://example.test/file", dest, expected_md5="0" * 32)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_retries_after_transient_failure_and_succeeds(tmp_path, monkeypatch):
    payload = b"retry succeeds" * 100
    calls = {"count": 0}

    def fake_get(url, stream, timeout, headers=None):
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
    def fake_get(url, stream, timeout, headers=None):
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
        lambda url, stream, timeout, headers=None: _TruncatedResponse(
            payload, declared_length=len(payload) + 100
        ),
    )
    dest = tmp_path / "out.bin"

    with pytest.raises(IncompleteDownloadError):
        download_file("http://example.test/file", dest)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_resumes_from_partial_file_after_midstream_failure(tmp_path, monkeypatch):
    """A retry after a mid-stream drop should send a Range request for only the
    missing bytes, rather than re-downloading (and re-verifying) the whole file."""
    payload = b"resume this download correctly across a dropped connection" * 50
    split = len(payload) // 3

    class _DropsMidStreamResponse(_FakeResponse):
        def iter_content(self, chunk_size):
            yield self._content[:split]
            raise requests.exceptions.ConnectionError("dropped mid-stream")

    calls = {"count": 0}

    def fake_get(url, stream, timeout, headers=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return _DropsMidStreamResponse(payload)
        assert headers == {"Range": f"bytes={split}-"}
        return _FakeResponse(payload[split:], status_code=206)

    monkeypatch.setattr("src.data.download.requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest)

    assert result == dest
    assert dest.read_bytes() == payload
    assert calls["count"] == 2


def test_download_restarts_from_scratch_if_server_ignores_range_header(
    tmp_path, monkeypatch
):
    """If the server responds 200 (full content) instead of 206 to a resume
    attempt, the partial file must be discarded and the download restarted
    rather than corrupting the file by appending full content after partial."""
    payload = b"server does not support byte ranges for this file" * 20
    split = 10

    class _DropsMidStreamResponse(_FakeResponse):
        def iter_content(self, chunk_size):
            yield self._content[:split]
            raise requests.exceptions.ConnectionError("dropped mid-stream")

    calls = {"count": 0}

    def fake_get(url, stream, timeout, headers=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return _DropsMidStreamResponse(payload)
        return _FakeResponse(payload, status_code=200)

    monkeypatch.setattr("src.data.download.requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest)

    assert result == dest
    assert dest.read_bytes() == payload
    assert calls["count"] == 2


def test_download_ignores_and_discards_stale_unrelated_part_file(tmp_path, monkeypatch):
    """A `.part` file left behind by an earlier, unrelated download (different
    URL, crashed process, etc.) must never be trusted as a resumable prefix —
    it should be discarded, not spliced into the new download's output."""
    payload = b"the actual, correct payload for this call" * 30
    dest = tmp_path / "out.bin"
    stale_part = dest.with_name(dest.name + ".part")
    stale_part.write_bytes(b"UNRELATED-STALE-DATA-FROM-A-DIFFERENT-DOWNLOAD" * 5)

    def fake_get(url, stream, timeout, headers=None):
        # A Range header here would mean the stale file was (wrongly) trusted.
        assert headers is None
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data.download.requests.get", fake_get)
    dest_result = download_file("http://example.test/file", dest)

    assert dest_result == dest
    assert dest.read_bytes() == payload


def test_download_cleans_up_part_file_on_non_request_exception(tmp_path, monkeypatch):
    """A failure that isn't a requests.exceptions.RequestException (e.g. a
    disk-full OSError while writing) must not leave the `.part` file behind
    for a later call to mistakenly resume from."""

    class _DropsWithOSErrorResponse(_FakeResponse):
        def iter_content(self, chunk_size):
            yield self._content[:5]
            raise OSError("disk full")

    monkeypatch.setattr(
        "src.data.download.requests.get",
        lambda url, stream, timeout, headers=None: _DropsWithOSErrorResponse(
            b"some payload"
        ),
    )
    dest = tmp_path / "out.bin"

    with pytest.raises(OSError):
        download_file("http://example.test/file", dest)

    assert not dest.with_name(dest.name + ".part").exists()


def test_download_discards_partial_file_when_resume_is_rejected_with_error_status(
    tmp_path, monkeypatch
):
    """If the server rejects a Range resume request with an error status
    (e.g. 416 Range Not Satisfiable) rather than ignoring it with 200, the
    partial file must still be discarded so a subsequent retry starts fresh
    instead of repeating the same doomed range request forever."""
    payload = b"eventually succeeds once the stale partial file is dropped" * 20
    split = 10

    class _DropsMidStreamResponse(_FakeResponse):
        def iter_content(self, chunk_size):
            yield self._content[:split]
            raise requests.exceptions.ConnectionError("dropped mid-stream")

    class _RangeNotSatisfiableResponse(_FakeResponse):
        def __init__(self):
            super().__init__(b"", status_code=416)

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("416 Range Not Satisfiable")

    calls = {"count": 0}

    def fake_get(url, stream, timeout, headers=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return _DropsMidStreamResponse(payload)
        if calls["count"] == 2:
            assert headers == {"Range": f"bytes={split}-"}
            return _RangeNotSatisfiableResponse()
        assert headers is None  # third attempt: partial file was dropped
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data.download.requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    dest = tmp_path / "out.bin"

    result = download_file("http://example.test/file", dest)

    assert result == dest
    assert dest.read_bytes() == payload
    assert calls["count"] == 3


def test_download_uses_connect_and_read_timeout_tuple(tmp_path, monkeypatch):
    payload = b"timeout tuple check"
    seen = {}

    def fake_get(url, stream, timeout, headers=None):
        seen["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data.download.requests.get", fake_get)
    dest = tmp_path / "out.bin"

    download_file("http://example.test/file", dest)

    assert seen["timeout"] == (15, 120)
