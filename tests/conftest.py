import pytest
import os
import cupy as cp


@pytest.fixture
def single_row() -> cp.array:
    return cp.arange(10, dtype=cp.int32)


@pytest.fixture
def single_weight() -> cp.array:
    return cp.array([1], dtype=cp.float32)


@pytest.fixture
def random_bin_matrix():
    cp.random.seed(7357)
    test_matrix = cp.random.randint(0, 2, (20, 50), dtype=cp.int32)
    return test_matrix


@pytest.fixture
def weight_array_ones():
    return cp.ones(20, dtype=cp.float32)


@pytest.fixture
def youchoose_raw_int() -> str:
    return os.path.join(os.path.dirname(__file__), 'data', 'youchoose_raw_int.csv')


@pytest.fixture
def youchoose_raw_str() -> str:
    return os.path.join(os.path.dirname(__file__), 'data', 'youchoose_raw_str.csv')


@pytest.fixture
def youchoose_preprocessed() -> str:
    return os.path.join(os.path.dirname(__file__), 'data', 'youchoose_preprocessed.csv')
