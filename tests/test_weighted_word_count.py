import pytest

weighted_wordcount_kernel = pytest.importorskip("vs_knn.weighted_word_count")


def test_wcount_formats(hello_str):
    print(hello_str)
