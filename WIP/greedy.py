import gc
import cudf
import cupy as cp
from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID
from cupyx.time import repeat

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
            int value_idx = target_values[col];
            //printf("row: %d, col: %d target_id: %d \n", row, col, value_idx);
            atomicAdd(&weighted_count[value_idx], weight[col]);
            //indices[idx[0]] = indices[idx[0]] + row;
            //atomicAdd(&idx[0], 1);
        }
   }
}
''', 'score_items_kernel')


def get_weighted_count(sessions, items, n_sessions, n_items):
    my_session = cp.array([1, 2, 3], dtype=cp.dtype('int32'))
    n_rows = len(sessions)
    n_cols = len(my_session)

    result_sessions = cp.zeros(n_sessions, dtype=cp.dtype('float32'))
    # result_items = cp.zeros(n_sessions, dtype=cp.dtype('float32'))

    n_blocks_x = int(n_rows / 64) + 1
    n_blocks_y = int(n_cols / 4) + 1

    # TODO: The number of threads per block should be a round multiple of the warp size,
    #  which is 32 on all current hardware.

    weights = cp.ones_like(my_session, dtype=cp.dtype('float32'))
    score_items_kernel(
        (n_blocks_x, n_blocks_y),
        (64, 4),
        (items, sessions, my_session, weights, n_rows, n_cols, result_sessions))

    return result_sessions


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

    n_sessions = len(cp.unique(session_tape))
    n_items = len(cp.unique(items_tape))

    # TODO:
    #  - Pre initialize the looong similarity vectors
    #  - Smaller vectors, but with indices to keep track of who writes where

    # sessions = random.choices(train_df[SESSION_ID].unique(), k=100)
    # session_items = [
    #     [int(e) for e in train_df[train_df[SESSION_ID] == session][ITEM_ID].values]
    #     for session in sessions
    # ]

    del train_df
    gc.collect()

    print(repeat(get_weighted_count, (session_tape, items_tape, n_sessions, n_items), n_repeat=10))

    a = 1
