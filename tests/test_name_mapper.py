import random
import cupy as cp
from vs_knn.name_mapper import NameIdxMap
from vs_knn.col_names import SESSION_ID, ITEM_ID


def test_name_mapper_int(youchoose_raw_int):
    convert_then_retrieve(youchoose_raw_int)


def test_name_mapper_str(youchoose_raw_str):
    convert_then_retrieve(youchoose_raw_str)


def convert_then_retrieve(df):
    nip = NameIdxMap().build(df)
    rand_gen = random.Random(73578173)

    random_query = rand_gen.choices(df[ITEM_ID].unique(), k=100)
    query_idx = nip.name_to_idx(random_query, ITEM_ID)
    query_names = nip.idx_to_name(query_idx, ITEM_ID)

    assert all([before == after for before, after in zip(random_query, query_names)])

    transformed_df = nip.get_transformed_df()
    item_idx_examples = cp.array(rand_gen.choices(transformed_df[ITEM_ID].unique(), k=5))
    sessions_idx = transformed_df[transformed_df[ITEM_ID].isin(item_idx_examples)][SESSION_ID].values
    session_names = nip.idx_to_name(sessions_idx, SESSION_ID)
    session_names_set = set(cp.asnumpy(session_names))

    original_df = df
    original_items = nip.idx_to_name(item_idx_examples, ITEM_ID)
    original_session_names = original_df[original_df[ITEM_ID].isin(original_items)][SESSION_ID].values
    original_session_names_set = set(cp.asnumpy(original_session_names))

    assert session_names_set == original_session_names_set


def test_save_name_mapper(tmpdir, youchoose_raw_int):
    nip = NameIdxMap().build(youchoose_raw_int)

    nip.save(tmpdir)

    nip2 = NameIdxMap()
    nip2.load(tmpdir)

    assert all(a == b for a, b in
               zip(nip.name_to_idx([1, 2, 2465], SESSION_ID), nip2.name_to_idx([1, 2, 2465], SESSION_ID)))
    assert all(a == b for a, b in
               zip(nip.idx_to_name([1, 2, 2465], SESSION_ID), nip2.idx_to_name([1, 2, 2465], SESSION_ID)))

