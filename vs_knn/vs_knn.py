import json
import pandas as pd
import cudf
from vs_knn.col_names import SESSION_ID, ITEM_ID, PI_I
from vs_knn.index_builder import IndexBuilder
from vs_knn.train_test_split import train_test_split


class VsKnnModel:
    def __init__(self, config_file='config.json', no_cudf=False):
        self.config_file = config_file

        with open('config.json', 'r') as f:
            project_config = json.load(f)
        self.config = project_config
        self.top_k = self.config.get('top_k', 100)
        self.index_builder = IndexBuilder(config_file=self.config_file, no_cudf=no_cudf)

        self.cudf = cudf
        if no_cudf:
            self.cudf = pd

    def train(self):
        self.index_builder.create_indices()

    def load(self):
        self.index_builder.load_indices()

    def train_test_split(self):
        train_test_split(config_file=self.config_file)

    def predict(self, query_items):
        query_df = self._step1_query_to_cudf(query_items)
        items_sessions_pi = self._step2_get_sessions_per_items(query_items, query_df)
        top_k_sessions = self._step3_keep_topk(items_sessions_pi)
        item_scores = self._step4_score_items_in_sessions(top_k_sessions)
        return item_scores

    def get_test_dict(self):
        test_series = pd.read_csv(
            self.config['data_sources']['test_data'],
            names=['sess', 'tms', 'item', 'cat'])
        processed_series = test_series.sort_values(by=['sess', 'tms'], ascending=[True, False]).drop(
            columns=['tms', 'cat'])
        session_to_item = processed_series.groupby('sess')['item'].apply(list)
        test_examples = pd.DataFrame(session_to_item).to_dict(orient='index')

        test_examples = {k: v['item'][0:self.config['items_per_session']] for k, v in test_examples.items()}
        return test_examples

    def _step1_query_to_cudf(self, session_items):
        """Convert session data to"""
        n_items = len(session_items)
        pi_i = [e / n_items for e in range(n_items, 0, -1)]
        session_df = self.cudf.DataFrame({'pi_i': pi_i},
                                         index=session_items)
        return session_df

    def _step2_get_sessions_per_items(self, query_items, query_df):
        """For each item in query get past sessions containing the item.
        Returns dataframe with item_id (index) corresponding session_id and pi_i value"""
        past_sessions = self.index_builder.item_index.loc[query_items]
        items_sessions_pi = past_sessions.join(query_df)
        return items_sessions_pi

    def _step3_keep_topk(self, df):
        df = df.groupby(SESSION_ID).agg({PI_I: 'sum'})
        return df.sort_values(by=[PI_I], ascending=False)[0:self.top_k]

    def _step4_score_items_in_sessions(self, top_k_sessions):
        """for the top k sessions with similarity scores, get the items in the sessions.
        Then get total similarity per item"""
        top_k_items = self.index_builder.session_index.loc[top_k_sessions.index]
        sessions_with_items = top_k_sessions.join(top_k_items)
        item_scores = sessions_with_items.groupby(ITEM_ID).agg({PI_I: 'sum'})
        return item_scores
