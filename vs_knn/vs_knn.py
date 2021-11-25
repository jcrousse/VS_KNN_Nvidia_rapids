import pandas as pd
import cudf
import cupy as cp
from vs_knn.col_names import SESSION_ID, ITEM_ID, PI_I
from vs_knn.index_builder import IndexBuilder
from vs_knn.weighted_word_count import weighted_word_count


class VsKnnModel:
    def __init__(self, project_config):
        self.config = project_config
        self.top_k = self.config.get('top_k', 100)

    def predict(self, query_items):
        raise NotImplementedError

    def _step1_ingest_query(self, session_items):
        raise NotImplementedError

    def get_session_similarities(self, query):
        raise NotImplementedError

    def _step3_keep_topk_sessions(self, session_items):
        raise NotImplementedError

    def get_item_similarities(self, sessions, session_similarities):
        raise NotImplementedError


class CupyVsKnnModel(VsKnnModel):
    def __init__(self, project_config, item_index, session_index, positional_weights):
        super().__init__(project_config)

        self.item_to_sessions = item_index
        self.session_to_items = session_index

        self.positional_weights = positional_weights

    def predict(self, query_items):
        sessions, session_similarities = self.get_session_similarities(query_items)
        if len(sessions) > self.top_k:
            sessions, session_similarities = self.keep_topk_sessions(sessions, session_similarities)
        unique_items, w_sum_items = self.get_item_similarities(sessions, session_similarities)
        return unique_items, w_sum_items

    def _step1_ingest_query(self, query_items):
        pass

    def get_session_similarities(self, query):
        item_slice = self.item_to_sessions[query]
        weights_slice = self.positional_weights[-len(query):]
        sessions, session_similarities = weighted_word_count(item_slice, weights_slice)
        return sessions, session_similarities

    def keep_topk_sessions(self, sessions, session_similarities):
        selection = cp.argsort(session_similarities)[0:self.top_k]
        return sessions[selection], session_similarities[selection]

    def get_item_similarities(self, sessions, session_similarities):
        session_slice = self.session_to_items[sessions]
        unique_items, w_sum_items = weighted_word_count(session_slice, session_similarities)
        return unique_items, w_sum_items

    def _step3_keep_topk_sessions(self, session_items):
        raise NotImplementedError


class DataframeVsKnnModel(VsKnnModel):
    def __init__(self, project_config, no_cudf=False):

        super().__init__(project_config)

        self.index_builder = IndexBuilder(project_config, no_cudf=no_cudf)

        self.cudf = cudf
        if no_cudf:
            self.cudf = pd

    def train(self):
        self.index_builder.create_indices()

    def load(self):
        self.index_builder.load_indices()

    def predict(self, query_items):
        query_df = self._step1_ingest_query(query_items)
        items_sessions_pi = self.get_session_similarities(query_df)
        top_k_sessions = self._step3_keep_topk_sessions(items_sessions_pi)
        item_scores = self._step4_get_item_similarities_df(top_k_sessions)
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

    def _step1_ingest_query(self, session_items):
        """Convert session data to"""
        n_items = len(session_items)
        pi_i = [e / n_items for e in range(n_items, 0, -1)]
        session_df = self.cudf.DataFrame({'pi_i': pi_i},
                                         index=session_items)
        return session_df

    def get_session_similarities(self, query_df):
        """For each item in query get past sessions containing the item.
        Returns dataframe with item_id (index) corresponding session_id and pi_i value"""
        past_sessions = self.index_builder.item_index.loc[query_df.index]
        items_sessions_pi = past_sessions.join(query_df)
        return items_sessions_pi

    def _step3_keep_topk_sessions(self, df):
        df = df.groupby(SESSION_ID).agg({PI_I: 'sum'})
        return df.sort_values(by=[PI_I], ascending=False)[0:self.top_k]

    def _step4_get_item_similarities_df(self, top_k_sessions):
        """for the top k sessions with similarity scores, get the items in the sessions.
        Then get total similarity per item"""
        top_k_items = self.index_builder.session_index.loc[top_k_sessions.index]
        sessions_with_items = top_k_sessions.join(top_k_items)
        item_scores = sessions_with_items.groupby(ITEM_ID).agg({PI_I: 'sum'})
        return item_scores

    def get_item_similarities(self, sessions, session_similarities):
        raise NotImplementedError
