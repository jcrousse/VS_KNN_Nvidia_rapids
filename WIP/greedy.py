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
        }
   }
}
''', 'score_items_kernel')


def get_weighted_count(sessions, items):
    my_session = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=cp.dtype('int32'))
    result = cp.zeros(len(sessions), dtype=cp.dtype('float32'))
    n_rows = len(sessions)
    n_cols = len(my_session)

    n_blocks_x = int(n_rows / 128) + 1
    n_blocks_y = int(n_cols / 8) + 1

    weights = cp.ones_like(my_session, dtype=cp.dtype('float32'))
    score_items_kernel(
        (n_blocks_x, n_blocks_y),
        (128, 8),
        (items, result, my_session, weights, n_rows, n_cols, result))
    return result


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

    # sessions = random.choices(train_df[SESSION_ID].unique(), k=100)
    # session_items = [
    #     [int(e) for e in train_df[train_df[SESSION_ID] == session][ITEM_ID].values]
    #     for session in sessions
    # ]

    del train_df
    gc.collect()

    print(repeat(get_weighted_count, (session_tape, items_tape), n_repeat=10))

    a = 1
