import random
import cudf
import cupy as cp
from vs_knn.col_names import SESSION_ID, ITEM_ID, TIMESTAMP, CATEGORY, YOUCHOOSE_COLUMNS
from vs_knn.name_mapper import NameIdxMap


def test_name_mapper_int(youchoose_raw_int):
    data_types = {SESSION_ID: cp.dtype('int32'), TIMESTAMP: cp.dtype('O'), ITEM_ID: cp.dtype('int32'),
                  CATEGORY: cp.dtype('int32')}

    df = cudf.read_csv(youchoose_raw_int,
                       names=YOUCHOOSE_COLUMNS,
                       dtype=data_types).sort_values(by=TIMESTAMP, ascending=False)

    convert_then_retrieve(df)


def test_name_mapper_str(youchoose_raw_str):
    data_types = {SESSION_ID: cp.dtype('str'), TIMESTAMP: cp.dtype('O'), ITEM_ID: cp.dtype('str'),
                  CATEGORY: cp.dtype('str')}

    df = cudf.read_csv(youchoose_raw_str,
                       names=YOUCHOOSE_COLUMNS,
                       dtype=data_types).sort_values(by=TIMESTAMP, ascending=False)

    convert_then_retrieve(df)


def convert_then_retrieve(df):
    nip = NameIdxMap().build(df)
    rand_gen = random.Random(73578173)

    random_query = rand_gen.choices(df[ITEM_ID].unique(), k=100)
    query_idx = nip.name_to_idx(random_query, ITEM_ID)
    query_names = nip.idx_to_name(query_idx, ITEM_ID)

    assert all([before == after for before, after in zip(random_query, query_names)])

    transformed_df = nip.get_transformed_df().to_pandas()
    item_idx_examples = rand_gen.choices(transformed_df[ITEM_ID].unique(), k=5)
    sessions_idx = list(transformed_df[transformed_df[ITEM_ID].isin(item_idx_examples)][SESSION_ID].values)
    session_names = set(nip.idx_to_name(sessions_idx, SESSION_ID))

    original_df = df.to_pandas()
    original_items = nip.idx_to_name(item_idx_examples, ITEM_ID)
    original_session_names = set(original_df[original_df[ITEM_ID].isin(original_items)][SESSION_ID].values)

    assert session_names == original_session_names

