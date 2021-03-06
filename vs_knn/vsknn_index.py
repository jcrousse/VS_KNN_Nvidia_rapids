import cupy as cp

from vs_knn.col_names import SESSION_ID, ITEM_ID


class OneDimVsknnIndex:
    """
    One Dimensional index, keeps a 1-D array in memory of values.
    Slower compute time, but lower memory usage.
    Suitable for querying a few elements at the same time: For items to sessions (as sessions are not expected
    to be very long)
    High level overview:
    - a 1D CuPy array 'values_array' that contains the session_id (item_id) column from the
    original dataset, grouped by item_id (session_id).
    - A 2D numpy array 'id_to_idx' contains as many rows as there are unique items (sessions), and three columns.
        At row i, the three columns give the starting index, ending index and length
        of the item i (session i) sessions (items)  in 'values_array'
    """
    def __init__(self):
        self.value_array: cp.array = None
        self.id_to_idx: cp.array = None

    def build_index(self, train_data, index_key=ITEM_ID, index_value=SESSION_ID):
        self.id_to_idx, self.value_array = self.build_idx_arrays(train_data, index_key, index_value)

        dmf = round((self.value_array.nbytes + self.id_to_idx.nbytes) / 10 ** 6, 2)
        print(f"Device memory footprint for index objects: {dmf} Mb ({index_key} index)")
        return self

    @staticmethod
    def build_idx_arrays(train_data, index_key=ITEM_ID, index_value=SESSION_ID, int_type=cp.intc):
        start_end_idx_df = train_data \
            .sort_values(by=index_key) \
            .reset_index().drop(columns="index") \
            .reset_index().rename(columns={"index": "end_idx"}) \
            .reset_index().rename(columns={"index": "start_idx"})

        value_array = start_end_idx_df[index_value].values.astype(int_type)

        key_table = start_end_idx_df. \
            groupby(index_key). \
            agg({"start_idx": "min", "end_idx": "max"}) \
            .sort_index()

        key_table["len"] = key_table["end_idx"] - key_table["start_idx"] + 1

        id_to_idx = key_table.values.astype(int_type)

        return id_to_idx, value_array

    def __getitem__(self, query):
        values_idx = self.id_to_idx[[int(q) for q in query]]
        max_len = int(values_idx[:, 2].max())
        values_slice = cp.vstack([
            cp.pad(self.value_array[cp.arange(int(l[0]), int(l[1]) + 1)], (0, max_len - int(l[2])))
            for l in values_idx
        ])
        return values_slice


class TwoDimVsknnIndex:
    """
    Two Dimensional index, keeps a 2-D array in memory of values.
    Row i contains the values for key i, padded with 0.
    Fast to use, but memory hungry (due to sparsity)
    Suitable for querying many elements at the same time: For sessions to items.
    High level overview:
    - a 2D CuPy array 'values_array' that contains the session_id (item_id) for item i (session i) at row i
    """

    def __init__(self):
        self.value_array: cp.array = None

    # todo: reshape in batches ?
    def build_index(self, train_data, index_key=SESSION_ID, index_value=ITEM_ID):
        train_data = train_data.sort_values(by=[index_key, index_value], ascending=[True, True])
        train_data = train_data.reset_index().drop('index', axis=1)
        train_data[ITEM_ID] = train_data[ITEM_ID].astype(cp.uintc)
        train_data[SESSION_ID] = train_data[SESSION_ID].astype(cp.uintc)
        cum_count_col = train_data.groupby(index_key).cumcount().astype(cp.uintc)
        train_data["position"] = cum_count_col
        reshaped_df = train_data.pivot(index=index_key, columns="position", values=[index_value])
        self.value_array = reshaped_df.fillna(0).values

        dmf = round(self.value_array.nbytes / 10 ** 6, 2)
        print(f"Device memory footprint for index objects: {dmf} Mb ({index_key} index)")
        return self

    def __getitem__(self, query):
        return self.value_array[query, :]

