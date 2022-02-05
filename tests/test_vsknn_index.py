import random
import cudf
import cupy as cp
from vs_knn.vsknn_index import OneDimVsknnIndex
from vs_knn.col_names import SESSION_ID, ITEM_ID, TIMESTAMP, PREPROCESSED_COLUMNS


def test_onedim_index(youchoose_preprocessed):
    rand_gen = random.Random(73578173)
    data_types = {SESSION_ID: cp.dtype('int32'), ITEM_ID: cp.dtype('int32')}

    df = cudf.read_csv(youchoose_preprocessed,
                       names=PREPROCESSED_COLUMNS,
                       dtype=data_types)

    test_index = OneDimVsknnIndex().build_index(df)

    item_idx_sample = rand_gen.choices(df[ITEM_ID].unique(), k=5)
    indexed_values = test_index[item_idx_sample]

    session_values = df[df[ITEM_ID].isin(item_idx_sample)][SESSION_ID].values
    bc1 = cp.bincount(indexed_values.flatten())
    bc2 = cp.bincount(session_values)
    bc1[0] = 0 # ignore padding
    assert all(bc1 == bc2)
