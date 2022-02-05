import random
import cudf
import cupy as cp
from vs_knn.vsknn_index import OneDimVsknnIndex, TwoDimVsknnIndex
from vs_knn.col_names import SESSION_ID, ITEM_ID, PREPROCESSED_COLUMNS


def test_onedim_index(youchoose_preprocessed):
    index_testing_function(youchoose_preprocessed, OneDimVsknnIndex, ITEM_ID, SESSION_ID)


def test_twodim_index(youchoose_preprocessed):
    index_testing_function(youchoose_preprocessed, TwoDimVsknnIndex, SESSION_ID, ITEM_ID)


def index_testing_function(data_path, index_object, key_col, val_col):
    rand_gen = random.Random(73578173)
    data_types = {SESSION_ID: cp.dtype('int32'), ITEM_ID: cp.dtype('int32')}

    df = cudf.read_csv(data_path,
                       names=PREPROCESSED_COLUMNS,
                       dtype=data_types)

    test_index = index_object().build_index(df, index_key=key_col, index_value=val_col)

    key_idx_sample = rand_gen.choices(df[key_col].unique(), k=5)
    indexed_values = test_index[key_idx_sample]

    expected_values = df[df[key_col].isin(key_idx_sample)][val_col].values
    bc1 = cp.bincount(indexed_values.flatten())
    bc2 = cp.bincount(expected_values)
    bc1[0] = 0  # ignore padding
    assert all(bc1 == bc2)

