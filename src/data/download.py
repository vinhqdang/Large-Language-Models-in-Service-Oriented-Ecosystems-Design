"""Streamed, checksum-verified file download."""
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's MD5 does not match the expected value."""


def download_file(
    url: str,
    dest_path: Path,
    expected_md5: str | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                md5.update(chunk)
                bar.update(len(chunk))

    if expected_md5 is not None and md5.hexdigest() != expected_md5:
        actual = md5.hexdigest()
        dest_path.unlink()
        raise ChecksumMismatchError(
            f"{dest_path}: expected MD5 {expected_md5}, got {actual}"
        )

    return dest_path
