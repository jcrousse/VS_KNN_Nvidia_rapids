import cudf
import pandas as pd
import cupy as cp
import gc
from vs_knn.data_read_write import read_dataset
from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID, ITEM_POSITION, CATEGORY


class IndexBuilder:
    def __init__(self, items_per_session=10, sessions_per_item=5000, no_cudf=False):

        self.items_per_session = items_per_session
        self.sessions_per_item = sessions_per_item

        self.session_index = None
        self.item_index = None

        self.cudf = cudf
        if no_cudf:
            self.cudf = pd

    def create_indices(self, train_df, max_sessions=None):

        if max_sessions:
            train_df = train_df[train_df[SESSION_ID] < max_sessions]
        self.session_index = self._top_items_per_sessions(train_df)
        self.item_index = self._top_sessions_per_items(train_df)

    def load_indices(self):
        # removed
        pass

    def save_indices(self):
        # removed
        pass

    def _top_items_per_sessions(self, df,):
        df = self._select_top_rows(df, self.items_per_session, SESSION_ID, TIMESTAMP, ITEM_ID)
        df = df.drop(columns=[TIMESTAMP])
        df = df.set_index(SESSION_ID)
        return df

    def _calculate_item_pos(self, df, ):
        df = df.sort_values(by=[SESSION_ID, TIMESTAMP], ascending=[True, True])
        df = df.reset_index()
        df = df.drop(columns='index')
        df[ITEM_POSITION] = df.groupby(SESSION_ID).cumcount()
        return df

    def _top_sessions_per_items(self, df, num_keep=5000, group_key='item_id'):
        df = self._select_top_rows(df, num_keep, ITEM_ID, SESSION_ID, TIMESTAMP)
        df = df.drop(columns=[TIMESTAMP])
        df = df.set_index(group_key)
        return df

    def _select_top_rows(self, df, num_keep, group_key, sort_col1, sort_col2):
        df = df.sort_values(by=[group_key, sort_col1, sort_col2], ascending=[True, True, False]).reset_index() \
            .drop(columns='index')
        df['cum_count'] = df.groupby(group_key).cumcount()
        df = df.drop_duplicates(subset=[group_key, sort_col1])
        df = df.loc[df.cum_count < num_keep]
        df = df.drop(columns=['cum_count'])
        return df

    def get_index_as_array(self, index='item'):
        reshaped_df = self._reshape_index(index)
        reshaped_df = self._expand_table(reshaped_df)
        reshaped_df = reshaped_df.fillna(0)
        return reshaped_df.values

    def _reshape_index(self, index='item'):
        idx_df = self.item_index
        idx_name, col_name = ITEM_ID, SESSION_ID
        if index == 'session':
            idx_df = self.session_index
            idx_name, col_name = SESSION_ID, ITEM_ID
        idx_df = idx_df.reset_index().drop_duplicates()
        idx_df = idx_df.sort_values(by=[idx_name, col_name], ascending=[True, True])
        idx_df = idx_df.reset_index()
        cum_count_col = idx_df.groupby(idx_name).cumcount().astype(int)
        idx_df[ITEM_POSITION] = cum_count_col
        reshaped_df = idx_df.pivot(index=idx_name, columns=ITEM_POSITION, values=[col_name])

        del idx_df, cum_count_col
        gc.collect()

        return reshaped_df

    @staticmethod
    def _expand_table(table):
        max_value = max(table.index.values)
        empty_sized_table = cudf.DataFrame(index=cp.arange(max_value))
        return empty_sized_table.join(table)

    def get_df_index(self, index='item', mode='cudf'):
        reshaped_df = self._reshape_index(index)
        return DataFrameIndex(reshaped_df, mode=mode)

    def get_unique_sessions(self):
        return self.session_index.index.unique().values

    def get_dict_index(self, index='item'):
        idx_df, target_size = self.item_index, self.sessions_per_item
        if index == 'session':
            idx_df, target_size = self.session_index, self.items_per_session

        renamed_df = idx_df.reset_index()
        renamed_df.columns = ['key', 'value']

        return DictIndex(renamed_df, target_size, index)


class DataFrameIndex:
    """
    slower than CuPy array as index, but less memory-hungry
    """
    def __init__(self, index_df: cudf.DataFrame, mode='cudf'):
        self.index_df = index_df.fillna(0)
        self.get_function = self._get_index_using_cudf
        if mode == 'pandas':
            self.index_df = self.index_df.to_pandas()
            self.get_function = self._get_index_using_pandas
        self.shape = self.index_df.shape

        self.known_items = set(self.index_df.index.values)

    def __getitem__(self, item):
        return self.get_function(item)

    def _get_index_using_pandas(self, item):
        item = item.tolist()
        return cp.array(self.index_df.loc[item, :].values)

    def _get_index_using_cudf(self, item):
        if item.size == 1:
            item = [item]
        return cp.array(self.index_df.loc[item, :].values)


def whatever_function(p_list):
    return p_list


def list_to_cp(p_list):
    return cp.pad(cp.array(p_list), (0, 10))


class DictIndex:
    def __init__(self, data_df, array_size, name):
        import pickle
        import os

        # stored_idx_pkl = name + '.plk'
        # if not os.path.isfile(stored_idx_pkl):

        array_per_key = data_df \
            .groupby('key') \
            .agg({'value': 'collect'}) \
            .to_pandas()\
            .to_dict(orient='index')

        processed_arrays = {k: cp.pad(cp.array(v['value']), (0, array_size))
                            for k, v in array_per_key.items()}
        # with open(stored_idx_pkl, 'w') as f:
        #     pickle.dump(processed_arrays, f)

        self.data_arrays = processed_arrays
        self.shape = (0, len(self.data_arrays))

    def __getitem__(self, item):
        # todo: something better than a list comprehension loop here
        return cp.vstack([self.data_arrays[e] for e in item.tolist()])
