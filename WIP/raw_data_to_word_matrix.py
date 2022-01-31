"""
Performance comparison between different methods to get all similar* sessions from a set of items,
given input raw data as two columns: session_id and item_id;

* similar = at least one item in common
"""

import cudf
import cupy as cp
from cupyx.time import repeat
import random

data_path = "data/train_set.dat"
col_names = ['session_id', 'timestamp', 'item_id', 'category']
cp_data = cudf.read_csv(data_path,  names=col_names, nrows=10000)[['session_id', 'timestamp',  'item_id']]\
    .sort_values(by='timestamp', ascending=False)\
    .drop_duplicates(subset=['session_id', 'item_id'], keep='first')

cp_data["item_id"] = "item_" + cp_data["item_id"].astype(str)

start_end_idx_df = cp_data\
    .sort_values(by="item_id")\
    .reset_index().drop(columns="index")\
    .reset_index().rename(columns={"index": "end_idx"}) \
    .reset_index().rename(columns={"index": "start_idx"})

sessions_per_item_cp = start_end_idx_df['session_id'].values

item_to_idx_df = start_end_idx_df.\
    groupby('item_id').\
    agg({"start_idx": "min", "end_idx": "max"})\
    .sort_index()

item_to_idx_cp = item_to_idx_df.values

itemid_to_itemidx = item_to_idx_df\
    .reset_index().rename(columns={"index": "item_id"})\
    .reset_index().set_index("item_id")['index']\
    .to_pandas().to_dict()

sessions = cp_data['session_id'].unique()

item_to_itemidx = cp_data[['item_id']] \
                    .reset_index() \
                    .set_index('item_id') \
                    .to_pandas()['item_id'].to_dict()

def random_session():
    random_session_id = random.choice(sessions)
    return cp_data[cp_data['session_id'] == random_session_id]['item_id'].values


def cudf_indexing():
    random_session_items = random_session()

    # still need to reshape, but already too slow....
    # For each item, array contains start/end indices of item (2 rows) in raw dataset
    # return session values at all indices
    return random_session_items


def cupy_indexing():
    random_session_items = random_session()
    sessions_scope = session_cp[(item_cp in random_session_items).any()]
    # no easy way to get index of items common to both array
    return sessions_scope



# def custom_kernel_indexing():
#     random_session_items = random_session()
#     found_indices = cp.zeros()
#     return index_match(
#         (n_blocks_x, n_blocks_y),
#         (64, 16),
#         (item_cp, random_session_items, found_indices)
#     )




if __name__ == '__main__':
    print(repeat(random_session, n_repeat=10))
    # print(repeat(cudf_indexing,  n_repeat=10))
    print(repeat(cupy_indexing, n_repeat=10))
