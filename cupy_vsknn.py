import cupy as cp
import numpy as np
from vs_knn.index_builder import IndexBuilder
from cupyx.time import repeat


weighted_wordcount_kernel = cp.RawKernel(r'''
extern "C" __global__
void weighted_wordcount_kernel(const int* sessions, const float* weight, const int* unique_sessions,  
                            const int n_unique_sessions, 
                            const int n_items, const int n_sessions,  float* y) {
                            
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int s_id = blockDim.z * blockIdx.z + threadIdx.z;

    if (row < n_items && col < n_sessions && s_id < n_unique_sessions){
        int session_id = sessions[(row * n_sessions + col)];
        if (session_id != -1 && session_id == unique_sessions[s_id]){
            atomicAdd(&y[s_id], weight[row]);
        }
   }
}
''', 'weighted_wordcount_kernel')


def num_count(relevant_slice, weights, current_session_len, sessions_per_items):
    n_blocks_x = int(relevant_slice.shape[1] / 16) + 1
    n_blocks_y = int(relevant_slice.shape[0] / 16) + 1
    unique_sessions = cp.unique(relevant_slice)
    if unique_sessions[0] == -1:
        unique_sessions = unique_sessions[1:]
    n_unique_sessions = len(unique_sessions)
    n_blocks_z = int(n_unique_sessions / 4) + 1
    weighted_sum = cp.zeros((n_unique_sessions,), dtype=np.float32)
    weighted_wordcount_kernel(
        (n_blocks_x, n_blocks_y, n_blocks_z),
        (16, 16, 4),
        (relevant_slice, weights, unique_sessions, n_unique_sessions, current_session_len, sessions_per_items, weighted_sum))
    return unique_sessions, weighted_sum


def vs_knn_predict(session, weights):
    item_slice = ITEM_TO_SESSIONS[session]
    unique_sessions, w_sum_sessions = num_count(item_slice, weights, len(session), SESSIONS_PER_ITEM)
    if unique_sessions:  # todo: some items have no historical sessions? 31, 287
        if len(unique_sessions) > K_SESSIONS:
            # sort and select top K
            print("todo: implement sort & select here")
        session_slice = SESSION_TO_ITEMS[unique_sessions]
        unique_items, w_sum_items = num_count(session_slice, w_sum_sessions, len(unique_sessions), ITEMS_PER_SESSION)
        return unique_items, w_sum_items


def run_random_test():
    random_session_id = cp.random.randint(0, MAX_SESSIONS)
    session = SESSION_TO_ITEMS[random_session_id]
    session_clean = session[cp.where(session > -1)]
    n_items = len(session_clean)
    items, scores = vs_knn_predict(session_clean, ITEM_WEIGHTS[-n_items:])
    return items, scores


index_builder = IndexBuilder()
MAX_SESSIONS = 100000  # saving memory
MAX_SESSION_LEN = 100
K_SESSIONS = 100
index_builder.create_indices('train_data', save=False, max_sessions=MAX_SESSIONS)

SESSION_TO_ITEMS = index_builder.get_index_as_table('session')
ITEM_TO_SESSIONS = index_builder.get_index_as_table('item')

ITEMS_PER_SESSION = SESSION_TO_ITEMS.shape[1]
SESSIONS_PER_ITEM = ITEM_TO_SESSIONS.shape[1]
ITEM_WEIGHTS = (cp.arange(MAX_SESSION_LEN, dtype=cp.float32) + 1) / MAX_SESSION_LEN

if __name__ == '__main__':
    print(repeat(run_random_test, n_repeat=1000))
