"""
Performance comparison between different methods to get all similar (= at least one item in common) sessions from a
set of items, given input raw data as three columns: session_id, item_id and timestamp;


"""

import cudf
import cupy as cp
import numpy as np
from cupyx.time import repeat
import random

ITEM_ID = "item_id"
SESSION_ID = "session_id"
TIMESTAMP = "timestamp"


class CupyIndex:
    """
    High level overview:
    - A python dictionary 'name_to_id' maps the item_id (session_id) in raw data to an int array index.
    As many entries as the number of items (sessions) in the dataset.
    This step is just to allow any hashable value for item_id, instead of assuming
    contiguous int item_ids (session_id) from 0 to N
    - a 1D CuPy array 'values_array' that contains the session_id (item_id) column from the
    original dataset, grouped by item_id (session_id).
    - A 2D numpy array 'id_to_idx' contains as many rows as there are unique items (sessions), and three columns.
        At row i, the three columns give the starting index, ending index and length
        of the item i (session i) sessions (items)  in 'values_array'
    """
    def __init__(self):
        self.name_to_id = {}
        self.value_array: cp.array = None
        self.id_to_idx: cp.array = None

    def build_index(self, train_data, index_key, index_value):

        start_end_idx_df = train_data \
            .sort_values(by=index_key) \
            .reset_index().drop(columns="index") \
            .reset_index().rename(columns={"index": "end_idx"}) \
            .reset_index().rename(columns={"index": "start_idx"})

        self.value_array = start_end_idx_df[index_value].values

        key_table = start_end_idx_df. \
            groupby(index_key). \
            agg({"start_idx": "min", "end_idx": "max"}) \
            .sort_index()

        key_table["len"] = key_table["end_idx"] - key_table["start_idx"]

        self.id_to_idx = key_table.values

        self.name_to_id = key_table \
            .reset_index().rename(columns={"index": index_key}) \
            .reset_index().set_index(index_key)['index'] \
            .to_pandas().to_dict()

        dmf = round((self.value_array.nbytes + self.id_to_idx.nbytes) / 10 ** 6, 2)
        print(f"Device memory footprint for index objects: {dmf} Mb")

    def __getitem__(self, query):
        query_ids = [self.name_to_id[e] for e in query]
        values_idx = self.id_to_idx[query_ids]
        max_len = int(values_idx[:, 2].max())
        values_slice = cp.vstack([
            cp.pad(self.value_array[cp.arange(int(l[0]), int(l[1]) + 1)], (0, max_len - int(l[2])))
            for l in values_idx
        ])
        return values_slice


class CudaVsKnn:

    def __init__(self, max_item_per_session=None):
        self.max_items_per_session = max_item_per_session

        self.item_to_sessions = CupyIndex()
        self.session_to_item = CupyIndex()

    def train(self, train_data: cudf.DataFrame):

        self._validate_data(train_data)

        if self.max_items_per_session:
            train_data = self._keep_n_latest_sessions(train_data)

        self.item_to_sessions.build_index(train_data, ITEM_ID, SESSION_ID)

    def predict(self, session):
        """"""
        historical_sessions = self.item_to_sessions[session]

    def _validate_data(self, train_data: cudf.DataFrame):
        """quick check that the expected columns are there (based on name)"""
        expected_columns = {ITEM_ID, SESSION_ID, TIMESTAMP}
        if expected_columns - set(train_data.columns):
            raise ValueError(f"Missing expected columns {expected_columns - set(train_data.columns)}")

    def _keep_n_latest_sessions(self, df):
        df = df.sort_values(by=['item_id', 'timestamp'], ascending=[True, False]).reset_index()
        df['session_n'] = df.groupby('item_id').cumcount()
        df = df[df['session_n'] <= self.max_items_per_session]
        return df[['item_id', 'session_id']]


def predict_random_session(model: CudaVsKnn, session_sample):
    """maybe do this with pandas stuff? doesn't matter if on GPU or not, we assume we
    start with a list of items"""
    random_session = random.choice(session_sample)
    model.predict(random_session)


if __name__ == '__main__':
    DEV = True  # False for full dataset
    nrows = 10000 if DEV else None
    data_path = "data/train_set.dat"
    col_names = ['session_id', 'timestamp', 'item_id', 'category']
    data_types = {'session_id': cp.dtype('int32'), 'timestamp': cp.dtype('O'), 'item_id': cp.dtype('int32'),
                  'category': cp.dtype('int32')}
    youchoose_data = cudf.read_csv(
        data_path,
        names=col_names,
        dtype=data_types,
        nrows=nrows)[['session_id', 'timestamp', 'item_id']] \
        .sort_values(by='timestamp', ascending=False)

    # also works with strings (can be tested by un-commenting line below
    # youchoose_data["item_id"] = "item_" + youchoose_data["item_id"].astype(str)

    n_rand_sessions = 200
    unique_sessions = youchoose_data['session_id'].unique()
    random_session_ids = random.choices(unique_sessions, k=n_rand_sessions)
    sessions_sample = youchoose_data[youchoose_data['session_id'].isin(random_session_ids)] \
        .groupby('session_id').agg({'item_id': 'collect'}).to_pandas()['item_id'].values


    vsk_model = CudaVsKnn()
    vsk_model.train(youchoose_data)

    print(repeat(predict_random_session, (vsk_model, sessions_sample), n_repeat=100))
    # print(repeat(cudf_indexing,  n_repeat=10))

