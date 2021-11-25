import pytest

weighted_wordcount_kernel = pytest.importorskip("cupy_vsknn")


def test_wcount_formats():
    print("hello")
