import pytest

weighted_wordcount_kernel = pytest.importorskip("cupy_vsknn")


# The VSKNN object receives a SESSION_TO_ITEM & ITEM_TO_SESSION which have __getintem__ function


def test_wcount_formats():
    print("hello")
