import json
import cudf
import pandas as pd
import cupy as cp
import gc

from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID, ITEM_POSITION


class IndexBuilder:
    def __init__(self, config_file='config.json', no_cudf=False):
        with open(config_file, 'r') as f:
            project_config = json.load(f)
        
        self.items_per_session = project_config['items_per_session']
        self.sessions_per_item = project_config['sessions_per_item']
        
        self.data_sources = project_config['data_sources']
        self.index_storage = project_config['index_storage']

        self.item_position_column = project_config['item_position_column']

        self.session_index = None
        self.item_index = None

        self.cudf = cudf
        if no_cudf:
            self.cudf = pd

    def create_indices(self, dataset='train_data', save=True, max_sessions=None):
        df = self.cudf.read_csv(self.data_sources[dataset],
                                names=[SESSION_ID, TIMESTAMP, ITEM_ID],
                                dtype={
                                    SESSION_ID: cp.dtype('int32'),
                                    TIMESTAMP: cp.dtype('O'),
                                    ITEM_ID: cp.dtype('int32')
                                },
                                usecols=[0, 1, 2]
                                )
        if max_sessions:
            df = df[df[SESSION_ID] < max_sessions]
        self.session_index = self._top_items_per_sessions(df)
        self.item_index = self._top_sessions_per_items(df)
        if save:
            self.save_indices()

    def load_indices(self):
        self.session_index = self.cudf.read_csv(self.index_storage['session_index'],
                                                index_col=SESSION_ID)
        self.item_index = self.cudf.read_csv(self.index_storage['item_index'],
                                             index_col=ITEM_ID)

    def save_indices(self):
        self.session_index.to_csv(self.index_storage['session_index'])
        self.item_index.to_csv(self.index_storage['item_index'])

    def _top_items_per_sessions(self, df,):
        df = self._select_top_rows(df, self.items_per_session, SESSION_ID, TIMESTAMP, ITEM_ID)
        if self.item_position_column:
            df = self._calculate_item_pos(df)
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

    def get_index_as_table(self, index='item'):
        idx_df = self.session_index
        idx_name, col_name = ITEM_ID, SESSION_ID
        if index == 'session':
            idx_df = self.item_index
            idx_name, col_name = SESSION_ID, ITEM_ID
        idx_df = idx_df.reset_index().drop_duplicates()
        idx_df = idx_df.sort_values(by=[idx_name, col_name], ascending=[True, True])
        idx_df = idx_df.reset_index()
        cum_count_col = idx_df.groupby(idx_name).cumcount().astype(int)
        idx_df[ITEM_POSITION] = cum_count_col
        reshaped_df = idx_df.pivot(index=idx_name, columns=ITEM_POSITION, values=[col_name])
        reshaped_df = self._expand_table(reshaped_df)
        del idx_df, cum_count_col
        gc.collect()
        reshaped_df = reshaped_df.fillna(-1)
        return reshaped_df.values

    @staticmethod
    def _expand_table(table):
        max_value = max(table.index.values)
        empty_sized_table = cudf.DataFrame(index=cp.arange(max_value))
        return empty_sized_table.join(table)

