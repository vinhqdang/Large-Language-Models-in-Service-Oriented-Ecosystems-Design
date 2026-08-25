import os
from pathlib import Path

import pytest

from src.data.paths import long_path


@pytest.mark.skipif(os.name != "nt", reason="Windows-only path behavior")
def test_long_path_uses_unc_prefix_for_unc_paths():
    result = long_path(Path(r"\\server\share\folder"))

    assert str(result).startswith(r"\\?\UNC\server\share\folder")
