"""
Performance comparison between different methods to get all similar (= at least one item in common) sessions from a
set of items, given input raw data as three columns: session_id, item_id and timestamp;

High level overview:
- A python dictionary 'itemid_to_itemidx' maps the item_id in raw data to an int array index. As many entries as the
number of items in the dataset
- a 1D CuPy array 'session_array' that contains session_ids, as many rows as the raw dataset.
- A 2D numpy array 'item_to_idx' contains as many rows as there are unique items, and three columns.
    At row i, the three columns give the starting index,
    ending index and length of the item i sessions in 'session_array'


"""

import cudf
import cupy as cp
import numpy as np
from cupyx.time import repeat
import random

data_path = "data/train_set.dat"
col_names = ['session_id', 'timestamp', 'item_id', 'category']
cp_data = cudf.read_csv(data_path,  names=col_names, nrows=10000)[['session_id', 'timestamp',  'item_id']]\
    .sort_values(by='timestamp', ascending=False)\
    .drop_duplicates(subset=['session_id', 'item_id'], keep='first')

# also works with strings (can be tested by un-commenting line below
# cp_data["item_id"] = "item_" + cp_data["item_id"].astype(str)

start_end_idx_df = cp_data\
    .sort_values(by="item_id")\
    .reset_index().drop(columns="index")\
    .reset_index().rename(columns={"index": "end_idx"}) \
    .reset_index().rename(columns={"index": "start_idx"})

session_array = start_end_idx_df['session_id'].values

item_to_idx_df = start_end_idx_df.\
    groupby('item_id').\
    agg({"start_idx": "min", "end_idx": "max"})\
    .sort_index()

item_to_idx_df["len"] = item_to_idx_df["end_idx"] - item_to_idx_df["start_idx"]

item_to_idx = cp.asnumpy(item_to_idx_df.values)

itemid_to_itemidx = item_to_idx_df\
    .reset_index().rename(columns={"index": "item_id"})\
    .reset_index().set_index("item_id")['index']\
    .to_pandas().to_dict()

n_rand_sessions = 200
unique_sessions = cp_data['session_id'].unique()
random_session_ids = random.choices(unique_sessions, k=n_rand_sessions)
sessions_sample = cp_data[cp_data['session_id'].isin(random_session_ids)]\
    .groupby('session_id').agg({'item_id': 'collect'}).to_pandas()['item_id'].values


def random_session():
    """maybe do this with pandas stuff? doesn't matter if on GPU or not, we assume we
    start with a list of items"""
    return random.choice(sessions_sample)


def cupy_indexing():
    random_session_items = random_session()
    items_idx = [itemid_to_itemidx[e] for e in random_session_items]
    session_location = item_to_idx[items_idx]
    max_len = session_location[:, 2].max()
    sessions = cp.vstack([
        cp.pad(session_array[np.arange(l[0], l[1] + 1)], (0, max_len - l[2]))
        for l in session_location
    ])
    return sessions


if __name__ == '__main__':
    print(repeat(random_session, n_repeat=100))
    # print(repeat(cudf_indexing,  n_repeat=10))
    print(repeat(cupy_indexing, n_repeat=100))
