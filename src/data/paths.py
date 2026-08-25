"""Windows long-path filesystem helper, shared across data-pipeline modules."""
import os
from pathlib import Path


def long_path(path: Path) -> Path:
    """On Windows, return a \\\\?\\-prefixed absolute path to bypass the 260-char
    MAX_PATH limit for filesystem calls. No-op on other platforms.

    UNC paths (\\\\server\\share\\...) require the distinct \\\\?\\UNC\\ prefix
    form rather than a plain \\\\?\\ prepended to the UNC path.
    """
    if os.name != "nt":
        return path
    resolved = path.resolve()
    resolved_str = str(resolved)
    if resolved_str.startswith("\\\\?\\"):
        return resolved
    if resolved_str.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{resolved_str.lstrip(chr(92))}")
    return Path(f"\\\\?\\{resolved_str}")
