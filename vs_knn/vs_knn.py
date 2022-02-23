import gc

import cudf
import cupy as cp
from vs_knn.col_names import SESSION_ID, ITEM_ID, TIMESTAMP
from vs_knn.vsknn_index import OneDimVsknnIndex
from vs_knn.name_mapper import NameIdxMap
from vs_knn.custom_kernels import copy_values_kernel, groubpy_kernel, copy_weights_kernel

int_type = cp.intc


class VsKnnModel:
    def __init__(self, top_k=100):
        self.top_k = top_k

    def train(self, train_df: cudf.DataFrame):
        raise NotImplementedError

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


def linear_decay(n):
    return (cp.arange(0, n, dtype=cp.float32) + 1) / n


def no_decay(n):
    return cp.ones(n, dtype=cp.float32)


class CupyVsKnnModel(VsKnnModel):
    def __init__(self,  decay='linear', top_k=100, max_sessions_per_items=5000, max_item_per_session=10):
        super().__init__(top_k)

        self.max_sessions_per_items = max_sessions_per_items
        self.max_items_per_session = max_item_per_session

        self._item_id_to_idx, self._item_values = cp.empty(1), cp.empty(1)
        self._sess_id_to_idx, self._sess_values = cp.empty(1), cp.empty(1)

        self._buffer_shape = max_sessions_per_items * max_item_per_session
        self._values_buffer = cp.zeros(self._buffer_shape, dtype=int_type)
        self._weights_buffer = cp.zeros(self._buffer_shape, dtype=cp.float32)

        self.name_map = NameIdxMap(skips_missings=True)

        if decay == 'linear':
            self.weight_function = linear_decay
        else:
            self.weight_function = no_decay

    def train(self, train_df: cudf.DataFrame, verbose=True):

        train_df = train_df.drop_duplicates(subset=[SESSION_ID, ITEM_ID], keep='first')

        if self.max_sessions_per_items:
            train_df = self._keep_n_latest_sessions(train_df)

        if self.max_items_per_session:
            train_df = self._keep_n_latest_items(train_df)

        train_df = train_df.drop(TIMESTAMP, axis=1)
        self.name_map = self.name_map.build(train_df)
        processed_df = self.name_map.get_transformed_df()

        del train_df
        gc.collect()

        self._item_id_to_idx, self._item_values = \
            OneDimVsknnIndex.build_idx_arrays(processed_df, ITEM_ID, SESSION_ID, int_type=int_type)
        self._sess_id_to_idx, self._sess_values = \
            OneDimVsknnIndex.build_idx_arrays(processed_df, SESSION_ID, ITEM_ID, int_type=int_type)

        # todo: change OneDimVsknnIndex so it does not add the third column in the first place
        self._item_id_to_idx = self._item_id_to_idx[:, 0:2]
        self._sess_id_to_idx = self._sess_id_to_idx[:, 0:2]

        del processed_df
        gc.collect()

        if verbose:
            index_arrays = [self._item_id_to_idx, self._item_values,
                            self._sess_id_to_idx, self._sess_values,
                            self._values_buffer, self._weights_buffer]
            total_bytes = sum([ia.nbytes for ia in index_arrays])
            dmf = round(total_bytes / 10 ** 6, 2)
            print(f"Device memory footprint for index objects: {dmf} Mb)")

    def predict(self, query_items):
        query_idx = self.name_map.name_to_idx(query_items, ITEM_ID)
        if query_idx:
            sessions, session_similarities = self.get_session_similarities(query_idx)
            if len(sessions) > self.top_k:
                sessions, session_similarities = self.keep_topk_sessions(sessions, session_similarities)
            unique_items, w_sum_items = self.get_item_similarities(sessions, session_similarities)
            if unique_items[0] == 0 and len(unique_items) > 1:
                unique_items, w_sum_items = unique_items[1:], w_sum_items[1:]
            ret_item_names = self.name_map.idx_to_name(unique_items, ITEM_ID)
        else:
            ret_item_names, w_sum_items = [], cp.array([])
        return ret_item_names, w_sum_items

    def _step1_ingest_query(self, query_items):
        pass

    def get_session_similarities(self, query):
        # todo: check not making deep copy here
        weights = self.weight_function(len(query))
        keys_array = self._item_id_to_idx[query]
        values_array = self._item_values
        sessions, session_similarities = self._get_similarities(keys_array, values_array, weights,
                                                                self.max_items_per_session)
        return sessions, session_similarities

    def keep_topk_sessions(self, sessions, session_similarities):
        selection = cp.argsort(session_similarities)[-self.top_k:]
        return sessions[selection], session_similarities[selection]

    def get_item_similarities(self, sessions, session_similarities):
        keys_array = self._sess_id_to_idx[sessions]
        values_array = self._sess_values
        unique_items, w_sum_items = self._get_similarities(keys_array, values_array, session_similarities,
                                                           self.max_items_per_session)
        return unique_items, w_sum_items

    def _get_similarities(self, key_array, values_array, weights_array, n_keys):
        self._copy_values_to_buffer(key_array, values_array, weights_array, n_keys)
        unique_values = cp.unique(self._values_buffer)
        similarities = self._reduce_buffer(unique_values)
        return unique_values, similarities

    def _copy_values_to_buffer(self, key_array, values_array, weights_array, n_keys):
        self._weights_buffer = cp.zeros(self._buffer_shape, dtype=cp.float32)
        n_values_per_keys = self.max_sessions_per_items if n_keys == self.max_items_per_session \
            else self.max_items_per_session
        kernel_args_v = (key_array, values_array,
                       len(key_array), n_values_per_keys, n_keys, self._buffer_shape,
                       self._values_buffer)
        kernel_args_w = (key_array, weights_array,
                       len(key_array), n_values_per_keys, n_keys, self._buffer_shape,
                       self._weights_buffer)
        t_per_block = 256
        target_threads = len(key_array) * n_values_per_keys
        n_blocks = int(target_threads / t_per_block) + 1
        copy_values_kernel((n_blocks,), (t_per_block,), kernel_args_v)
        # print("_______________new weights_____________")
        copy_weights_kernel((n_blocks,), (t_per_block,), kernel_args_w)

    def _reduce_buffer(self, unique_values):
        out_weights_groupby = cp.zeros_like(unique_values, dtype=cp.float32)
        n_items = self._buffer_shape * len(unique_values)
        kernel_args = (self._values_buffer, self._weights_buffer, unique_values,
                       len(unique_values), n_items,
                       out_weights_groupby)
        t_per_block = 256
        n_blocks = int(n_items / t_per_block) + 1
        groubpy_kernel((n_blocks,), (t_per_block,), kernel_args)
        return out_weights_groupby

    def _step3_keep_topk_sessions(self, session_items):
        raise NotImplementedError

    def _keep_n_latest_sessions(self, train_data):
        return self._keep_n_latest_values(train_data, ITEM_ID, self.max_sessions_per_items)

    def _keep_n_latest_items(self, train_data):
        return self._keep_n_latest_values(train_data, SESSION_ID, self.max_items_per_session)

    @staticmethod
    def _keep_n_latest_values(df, sort_key, n_keep):
        df = df.sort_values(by=[sort_key, TIMESTAMP], ascending=[True, False]).reset_index()
        df['value_n'] = df.groupby(sort_key).cumcount()
        df = df[df['value_n'] <= n_keep]
        return df.drop(['index', 'value_n'], axis=1)
