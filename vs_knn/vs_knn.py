import gc
import os
import pickle
import time
import asyncio

import cupy as cp
from vs_knn.col_names import SESSION_ID, ITEM_ID, TIMESTAMP
from vs_knn.vsknn_index import OneDimVsknnIndex
from vs_knn.name_mapper import NameIdxMap
from vs_knn.custom_kernels import copy_values_kernel, groubpy_kernel

int_type = cp.intc


def linear_decay(n):
    return (cp.arange(0, n, dtype=cp.float32) + 1) / n


def quadratic_decay(n):
    num = cp.arange(1, n + 1, dtype=cp.float32)
    denom = num * num
    return cp.flip(num / denom)


def no_decay(n):
    return cp.ones(n, dtype=cp.float32)


class CupyVsKnnModel:
    def __init__(self,  decay='linear', top_k=100, max_sessions_per_items=5000, max_item_per_session=10,
                 item_col=ITEM_ID, time_col=TIMESTAMP, session_col=SESSION_ID, waste_some_time=False):
        self.top_k = top_k

        self.max_sessions_per_items = max_sessions_per_items
        self.max_items_per_session = max_item_per_session

        self._item_id_to_idx, self._item_values = cp.empty(1), cp.empty(1)
        self._sess_id_to_idx, self._sess_values = cp.empty(1), cp.empty(1)

        self._buffer_shape = max_sessions_per_items * max_item_per_session

        self.item_col, self.time_col, self.session_col = item_col, time_col, session_col

        self.name_map = NameIdxMap(skips_missings=True)

        self.waste_some_time = waste_some_time

        if decay == 'linear':
            self.weight_function = linear_decay
        elif decay == 'quadratic':
            self.weight_function = quadratic_decay
        else:
            self.weight_function = no_decay

    def train(self, train_df, verbose=True):

        train_df = train_df.rename(columns={self.item_col: ITEM_ID,
                                            self.time_col: TIMESTAMP, self.session_col: SESSION_ID})

        train_df = train_df.drop_duplicates(subset=[SESSION_ID, ITEM_ID], keep='first')

        if self.max_sessions_per_items:
            train_df = self._keep_n_latest_sessions(train_df)

        if self.max_items_per_session:
            train_df = self._keep_n_latest_items(train_df)

        train_df = train_df.drop(TIMESTAMP, axis=1)
        self.name_map = self.name_map.build(train_df)
        processed_df = self.name_map.get_transformed_df()
        self.name_map.del_transformed_df()

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
                            ]
            total_bytes = sum([ia.nbytes for ia in index_arrays])
            dmf = round(total_bytes / 10 ** 6, 2)
            print(f"Device memory footprint for index objects: {dmf} Mb)")

    async def predict(self, queries: list):
        return_data = {
            'predicted_items': [],
            'scores': cp.array([]),
            'cpu_time': 0,
            'gpu_time': 0
        }
        start = time.time()
        stream = cp.cuda.Stream(non_blocking=True)
        stream.use()
        values_buffer = cp.zeros((len(queries), self._buffer_shape), dtype=int_type)
        weights_buffer = cp.zeros((len(queries), self._buffer_shape), dtype=cp.float32)
        queries_py = self.name_map.name_to_idx(queries, ITEM_ID)
        query_idx = cp.vstack([cp.pad(cp.array(q, dtype=int_type), (0, self.max_items_per_session - len(q)))
                               for q in queries_py])

        sessions, session_similarities = self.get_session_similarities(query_idx, values_buffer, weights_buffer)
        sessions, session_similarities = self.keep_topk_sessions(sessions, session_similarities)
        unique_items, w_sum_items = self.get_item_similarities(sessions, session_similarities, values_buffer,
                                                               weights_buffer)
        if self.waste_some_time:
            d_mat = cp.random.randn(1024 * 1024, dtype=cp.float64).reshape(1024, 1024)
            d_ret = d_mat
            for i in range(15):
                d_ret = cp.matmul(d_ret, d_mat)
        pre_synch = time.time()
        while not stream.done:
            await asyncio.sleep(0.0005)
        stream.synchronize()
        synch_time = time.time() - pre_synch
        return_data['predicted_items'] = self.name_map.idx_to_name(unique_items, ITEM_ID)
        return_data['scores'] = w_sum_items
        return_data['cpu_time'] = pre_synch - start
        return_data['gpu_time'] = synch_time
        return return_data

    def get_session_similarities(self, query, values_buffer, weights_buffer):
        weights = self.weight_function(self.max_items_per_session)
        # self._item_id_to_idx[cp.array(qq).flatten()].reshape(qq.shape)
        keys_array = self._item_id_to_idx[query]
        values_array = self._item_values
        sessions, session_similarities = self._get_similarities(keys_array, values_array, weights,
                                                                self.max_sessions_per_items,
                                                                values_buffer, weights_buffer)
        return sessions, session_similarities

    def keep_topk_sessions(self, sessions, session_similarities):
        selection = cp.argsort(session_similarities)[:, -self.top_k:]
        return cp.take_along_axis(sessions, selection, 1),  cp.take_along_axis(session_similarities, selection, 1)

    def get_item_similarities(self, sessions, session_similarities, vb, wb):
        keys_array = self._sess_id_to_idx[sessions]
        values_array = self._sess_values
        unique_items, w_sum_items = self._get_similarities(keys_array, values_array, session_similarities,

                                                           self.max_items_per_session, vb, wb)
        return unique_items, w_sum_items

    def _get_similarities(self, key_array, values_array, weights_array, n_keys, values_buffer, weights_buffer):
        self._copy_values_to_buffer(key_array, values_array, weights_array,
                                    n_keys, values_buffer, weights_buffer)
        arrays_unique = [cp.unique(values_buffer[i, :]) for i in range(len(key_array))]
        target_width = max([len(arr) for arr in arrays_unique])
        unique_values = cp.vstack(
            [cp.pad(arr, (0, target_width - len(arr))) for arr in arrays_unique]
        )
        similarities = self._reduce_buffer(unique_values, values_buffer, weights_buffer)
        return unique_values, similarities

    def _copy_values_to_buffer(self, key_array, values_array, weights_array, n_keys, values_buffer, weights_buffer):
        n_values_per_keys = self.max_sessions_per_items if n_keys == self.max_items_per_session \
            else self.max_items_per_session
        kernel_args = (key_array, values_array, weights_array,
                       key_array.shape[1], n_values_per_keys, n_keys, self._buffer_shape, key_array.shape[0],
                       values_buffer, weights_buffer)
        t_per_block = 256
        target_threads = len(key_array) * n_values_per_keys * n_keys
        n_blocks = int(target_threads / t_per_block) + 1
        copy_values_kernel((n_blocks,), (t_per_block,), kernel_args)

    def _reduce_buffer(self, unique_values, values_buffer, weights_buffer):
        out_weights_groupby = cp.zeros_like(unique_values, dtype=cp.float32)
        kernel_args = (values_buffer, weights_buffer, unique_values,
                       unique_values.shape[1], values_buffer.shape[1], values_buffer.shape[0],
                       out_weights_groupby)
        t_per_block = 256
        n_items = unique_values.shape[1] * values_buffer.shape[1] * values_buffer.shape[0]
        n_blocks = int(n_items / t_per_block) + 1
        groubpy_kernel((n_blocks,), (t_per_block,), kernel_args)
        return out_weights_groupby

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

    def save(self, dirname):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        cupy_store = os.path.join(dirname, '_cupy.npz')
        param_store = os.path.join(dirname, '_model_params.pkl')

        cp.savez(cupy_store, iid=self._item_id_to_idx, iv=self._item_values, sid=self._sess_id_to_idx,
                 sv=self._sess_values)

        with open(param_store, 'wb') as f:
            pickle.dump({
                'top_k': self.top_k,
                'max_sessions_per_items': self.max_sessions_per_items,
                'max_items_per_session': self.max_items_per_session,
                'name_map': self.name_map,
                'weight_function': self.weight_function
            }, f)

    def load(self, dirname):

        cupy_store = os.path.join(dirname, '_cupy.npz')
        param_store = os.path.join(dirname, '_model_params.pkl')

        saved_data = cp.load(cupy_store)
        self._item_id_to_idx = saved_data['iid']
        self._item_values = saved_data['iv']
        self._sess_id_to_idx = saved_data['sid']
        self._sess_values = saved_data['sv']

        with open(param_store, 'rb') as f:
            stored_params = pickle.load(f)
            self.top_k = stored_params["top_k"]
            self.max_sessions_per_items = stored_params["max_sessions_per_items"]
            self.max_items_per_session = stored_params["max_items_per_session"]
            self.name_map = stored_params["name_map"]
            self.weight_function = stored_params["weight_function"]
