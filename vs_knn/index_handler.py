import json
import cudf


class IndexHander:
    def __init__(self, config_file='config.json'):
        with open(config_file, 'r') as f:
            project_config = json.load(f)
        
        self.items_per_session = project_config['items_per_session']
        self.sessions_per_item = project_config['sessions_per_item']
        
        self.session_id = project_config['df_columns']['session_id']
        self.timestamp = project_config['df_columns']['timestamp']
        self.item_id = project_config['df_columns']['item_id']
        self.category = project_config['df_columns']['category']
        self.item_value = project_config['df_columns']['item_value']
        
        self.data_sources = project_config['data_sources']
        self.index_storage = project_config['index_storage']

        self.session_index = None
        self.item_index = None

    def create_indices(self, dataset='train', save=True):
        df = cudf.read_csv(self.data_sources[dataset],
                           names=[self.session_id, self.timestamp, self.item_id, self.category])
        self.session_index = self._top_items_per_sessions(df)
        self.item_index = self._top_sessions_per_items(df)
        self.save_indices()

    def load_indices(self):
        self.session_index = cudf.read_csv(self.index_storage['session_index'],
                                           index_col=self.session_id)
        self.item_index = cudf.read_csv(self.index_storage['item_index'],
                                        index_col=self.item_id)

    def save_indices(self):
        self.session_index.to_csv(self.index_storage['session_index'])
        self.item_index.to_csv(self.index_storage['item_index'])

    def _top_items_per_sessions(self, df, num_keep=100):
        df = self._select_top_rows(df, num_keep, self.session_id, self.timestamp, self.item_id)
        df = self._calculate_item_val(df)
        df = df.drop(columns=[self.timestamp])
        return df

    def _calculate_item_val(self, df, ):
        df = df.sort_values(by=[self.session_id, self.timestamp], ascending=[True, True]).reset_index().drop(columns='index')
        df[self.item_value] = df.groupby(self.session_id).cumcount()
        return df

    def _top_sessions_per_items(self, df, num_keep=5000, group_key='item_id'):
        df = self._select_top_rows(df, num_keep, self.item_id, self.session_id, self.timestamp)
        df = df.drop(columns=[self.timestamp])
        df = df.set_index(group_key)
        return df

    def _select_top_rows(self, df, num_keep, group_key, sort_col1, sort_col2):
        df = df.sort_values(by=[group_key, sort_col1, sort_col2], ascending=[True, True, False]).reset_index() \
            .drop(columns='index')
        df['cum_count'] = df.groupby(group_key).cumcount()
        df = df.drop_duplicates(subset=[group_key, sort_col1])
        df = df.loc[df.cum_count < num_keep]
        df = df.drop(columns=[self.category, 'cum_count'])
        return df
        

