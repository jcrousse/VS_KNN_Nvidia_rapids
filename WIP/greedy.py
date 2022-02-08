import gc
import cudf
import cupy as cp
from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID
from cupyx.time import repeat
import os
import pickle
import random

score_items_kernel = cp.RawKernel(r'''
extern "C" __global__
void score_items_kernel(
                            const int* keys_array, 
                            const int* values_array, 
                            const int* target_values,
                            const float* weight, 
                            const int num_rows, 
                            const int num_cols,
                            float* weighted_count) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    

    if (row < num_rows && col < num_cols){
    //printf("row: %d", row);
        if (keys_array[row] == target_values[col]){
            int value_idx = values_array[row];
            //printf("row: %d, col: %d session_id: %d\n", row, col, value_idx);
            atomicAdd(&weighted_count[value_idx], weight[col]);
            //indices[idx[0]] = indices[idx[0]] + row;
            //atomicAdd(&idx[0], 1);
        }
   }
}
''', 'score_items_kernel')


def get_test_sessions(df):
    stored_sessions_file = '../train_session.pkl'
    if not os.path.isfile(stored_sessions_file):
        random.seed(674837438)
        sessions = random.choices(df[SESSION_ID].unique(), k=1000)
        ret = [
            [int(e) for e in df[df[SESSION_ID] == session][ITEM_ID].values]
            for session in sessions
        ]
        with open(stored_sessions_file, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with open(stored_sessions_file, 'rb') as f:
            ret = pickle.load(f)
    return ret


def get_weighted_count(list_of_sessions, sessions, items, session_similarity):
    """
    Core: About 2ms
    Reset similarity vector: about 1ms
    From sparse to dense: about 2ms

    :param list_of_sessions:
    :param sessions:
    :param items:
    :param session_similarity:
    :return:
    """
    my_session = cp.array(list_of_sessions[random.randint(0, 999)], dtype=cp.dtype('int32'))
    # my_session = cp.array([1, 2], dtype=cp.dtype('int32'))
    n_rows = len(sessions)
    n_cols = len(my_session)

    session_similarity = session_similarity * 0

    n_blocks_x = int(n_rows / 64) + 1
    n_blocks_y = int(n_cols / 4) + 1

    weights = cp.ones_like(my_session, dtype=cp.dtype('float32'))
    score_items_kernel(
        (n_blocks_x, n_blocks_y),
        (64, 4),
        (items, sessions, my_session, weights, n_rows, n_cols, session_similarity))
    sessions_indices_dense = cp.nonzero(session_similarity)
    session_similarities_dense = session_tape[sessions_indices_dense]
    return sessions_indices_dense, session_similarities_dense


if __name__ == '__main__':
    train_data_path = "../data/train_set.dat"
    train_df = cudf.read_csv(train_data_path,
                             usecols=[0, 1, 2],
                             names=[SESSION_ID, TIMESTAMP, ITEM_ID],
                             dtype={
                                 SESSION_ID: cp.dtype('int32'),
                                 ITEM_ID: cp.dtype('int32'),
                                 TIMESTAMP: cp.dtype('O')
                             }
                             )

    session_tape = train_df[SESSION_ID].values
    items_tape = train_df[ITEM_ID].values

    session_similarity = cp.zeros(len(cp.unique(session_tape)), dtype=cp.dtype('float32'))
    item_similarity = cp.zeros(len(cp.unique(items_tape)), dtype=cp.dtype('float32'))

    # TODO:
    #  - Pre initialize the looong similarity vectors
    #  - Smaller vectors, but with indices to keep track of who writes where
    #  - Use cupy.nonzero to find sessions/items and their weight
    #  - Benchmark for different data set sizes. How does it scale with data

    # sessions = random.choices(train_df[SESSION_ID].unique(), k=100)
    # session_items = [
    #     [int(e) for e in train_df[train_df[SESSION_ID] == session][ITEM_ID].values]
    #     for session in sessions
    # ]


    # del train_df
    # gc.collect()

    for idx in range(5 * 10 ** 6, 40 * 10 ** 6, 5 * 10 ** 6):
        session_items = get_test_sessions(train_df)
        session_tape_subset = session_tape[0:idx]
        items_tape_subset = items_tape[0:idx]
        session_similarity = cp.zeros(len(cp.unique(session_tape_subset)), dtype=cp.dtype('float32'))
        item_similarity = cp.zeros(len(cp.unique(items_tape_subset)), dtype=cp.dtype('float32'))
        print(repeat(get_weighted_count, (session_items, session_tape_subset, items_tape_subset, session_similarity), n_repeat=100))


