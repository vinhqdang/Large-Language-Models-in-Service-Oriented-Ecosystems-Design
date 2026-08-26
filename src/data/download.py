"""Streamed, checksum-verified file download."""
import hashlib
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

MAX_ATTEMPTS = 8
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's MD5 does not match the expected value."""


class IncompleteDownloadError(RuntimeError):
    """Raised when fewer bytes are streamed than the server's content-length header
    declared, indicating a truncated download."""


def _hash_existing_file(temp_path: Path):
    md5 = hashlib.md5()
    if not temp_path.exists():
        return 0, md5
    with open(temp_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return temp_path.stat().st_size, md5


def _stream_to_temp(
    url: str,
    temp_path: Path,
    chunk_size: int,
):
    """Stream url into temp_path, resuming from any bytes already on disk.

    A prior attempt (within the same download_file call) may have left a
    partial temp_path behind. Rather than discarding that progress, this
    requests the remaining bytes via a Range header — important for large
    files over a connection that drops mid-stream, where restarting from
    byte 0 every retry may never finish. If the server doesn't honor the
    Range request with a 206 — whether it ignores it (200, full content) or
    rejects it outright (e.g. 416) — the partial file is discarded so the
    next attempt starts clean instead of repeating a doomed range request.
    """
    resume_from, md5 = _hash_existing_file(temp_path)
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else None

    with requests.get(
        url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    ) as response:
        if resume_from and response.status_code != 206:
            if temp_path.exists():
                temp_path.unlink()
            resume_from, md5 = 0, hashlib.md5()
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        remaining = int(content_length) if content_length is not None else None
        expected_total = resume_from + remaining if remaining is not None else None

        bytes_written = resume_from
        mode = "ab" if resume_from else "wb"
        with open(temp_path, mode) as f, tqdm(
            total=expected_total,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            desc=temp_path.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                md5.update(chunk)
                bytes_written += len(chunk)
                bar.update(len(chunk))

    return md5, bytes_written, expected_total


def download_file(
    url: str,
    dest_path: Path,
    expected_md5: str | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(dest_path.name + ".part")
    # Only a temp_path created by this call's own retry loop (below) is ever
    # trusted as a resumable prefix. Anything already there — a leftover from
    # a crashed prior run, or from a different URL that happens to resolve to
    # the same dest_path — is discarded up front rather than silently spliced
    # into the output.
    if temp_path.exists():
        temp_path.unlink()

    try:
        last_exc: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                md5, bytes_written, total = _stream_to_temp(url, temp_path, chunk_size)
                last_exc = None
                break
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(min(2**attempt, 30))

        if last_exc is not None:
            raise last_exc

        if total is not None and bytes_written != total:
            raise IncompleteDownloadError(
                f"{dest_path}: expected {total} bytes, got {bytes_written}"
            )

        if expected_md5 is not None and md5.hexdigest() != expected_md5:
            actual = md5.hexdigest()
            raise ChecksumMismatchError(
                f"{dest_path}: expected MD5 {expected_md5}, got {actual}"
            )
    except Exception:
        # Cover every failure path uniformly, not just the ones the retry
        # loop itself understands (requests.exceptions.RequestException) —
        # an unrelated crash (e.g. disk full) must not leave a partial file
        # behind for some later, unrelated download_file() call to trust.
        if temp_path.exists():
            temp_path.unlink()
        raise

    os.replace(temp_path, dest_path)
    return dest_path
