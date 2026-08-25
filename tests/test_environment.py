import sys

def test_python_version_is_313_or_newer():
    assert sys.version_info >= (3, 13), (
        f"Expected Python 3.13+, got {sys.version_info.major}.{sys.version_info.minor}"
    )

def test_src_package_importable():
    import src  # noqa: F401
    import src.data  # noqa: F401
