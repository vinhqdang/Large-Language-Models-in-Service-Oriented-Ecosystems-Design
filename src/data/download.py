"""Streamed, checksum-verified file download."""
import hashlib
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

MAX_ATTEMPTS = 3


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's MD5 does not match the expected value."""


class IncompleteDownloadError(RuntimeError):
    """Raised when fewer bytes are streamed than the server's content-length header
    declared, indicating a truncated download."""


def _stream_to_temp(
    url: str,
    temp_path: Path,
    chunk_size: int,
):
    md5 = hashlib.md5()
    bytes_written = 0

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        expected_total = int(content_length) if content_length is not None else None
        with open(temp_path, "wb") as f, tqdm(
            total=expected_total, unit="B", unit_scale=True, desc=temp_path.name
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

    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        if temp_path.exists():
            temp_path.unlink()
        try:
            md5, bytes_written, total = _stream_to_temp(url, temp_path, chunk_size)
            last_exc = None
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)

    if last_exc is not None:
        raise last_exc

    if total is not None and bytes_written != total:
        temp_path.unlink()
        raise IncompleteDownloadError(
            f"{dest_path}: expected {total} bytes, got {bytes_written}"
        )

    if expected_md5 is not None and md5.hexdigest() != expected_md5:
        actual = md5.hexdigest()
        temp_path.unlink()
        raise ChecksumMismatchError(
            f"{dest_path}: expected MD5 {expected_md5}, got {actual}"
        )

    os.replace(temp_path, dest_path)
    return dest_path
