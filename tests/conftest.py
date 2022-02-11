import pytest
import os
import cupy as cp
import cudf
from vs_knn.col_names import SESSION_ID, ITEM_ID, TIMESTAMP

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


@pytest.fixture
def tiny_vsknn_df() -> cudf.DataFrame:
    """
    Test: User session is [6, 4, 5], k = 2, nearest sessions should be the first 2 below.
    suggested items should be 3, 1, 2 in that order.
    """
    sessions = [
        [1, 3, 4, 5],
        [2, 3, 4, 5],
        [3, 4, 5, 1, 2],
        [4, 5, 1, 2, 3],
        [5, 1, 2, 3, 1]
    ]
    session_ids = flatten2d([['sess' + str(i)] * len(e) for i, e in enumerate(sessions)])
    timestamps = flatten2d([list(range(len(e))) for e in sessions])
    item_ids = flatten2d(sessions)

    df = cudf.DataFrame(data={
        SESSION_ID: session_ids,
        TIMESTAMP: timestamps,
        ITEM_ID: item_ids
    })

    return df


@pytest.fixture
def tiny_session():
    return [6, 4, 5]


def flatten2d(matrix):
    return [e for row in matrix for e in row]
