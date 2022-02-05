import random
import cudf
import cupy as cp
from vs_knn.col_names import SESSION_ID, ITEM_ID, TIMESTAMP, CATEGORY, YOUCHOOSE_COLUMNS
from vs_knn.name_mapper import NameIdxMap


def test_name_mapper_int(youchoose_raw_int):
    data_types = {SESSION_ID: cp.dtype('int32'), TIMESTAMP: cp.dtype('O'), ITEM_ID: cp.dtype('int32'),
                  CATEGORY: cp.dtype('int32')}

    df = cudf.read_csv(youchoose_raw_int,
                       col_names=YOUCHOOSE_COLUMNS,
                       dtype=data_types).sort_values(by='timestamp', ascending=False)

    convert_then_retrieve(df)


def test_name_mapper_str(youchoose_raw_str):
    data_types = {SESSION_ID: cp.dtype('str'), TIMESTAMP: cp.dtype('O'), ITEM_ID: cp.dtype('str'),
                  CATEGORY: cp.dtype('str')}

    df = cudf.read_csv(youchoose_raw_str,
                       col_names=YOUCHOOSE_COLUMNS,
                       dtype=data_types).sort_values(by='timestamp', ascending=False)

    convert_then_retrieve(df)


def convert_then_retrieve(df):
    nip = NameIdxMap().build(df)
    rand_gen = random.Random(73578173)

    random_query = rand_gen.choices(df[ITEM_ID].unique(), k=100)
    query_idx = nip.name_to_idx(random_query, ITEM_ID)
    query_names = nip.idx_to_name(query_idx, ITEM_ID)

    assert all([before == after for before, after in zip(random_query, query_names)])
