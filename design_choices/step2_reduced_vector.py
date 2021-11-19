import cupy as cp
import numpy as np
from cupyx.time import repeat

"""
Fast, 
but the time taken grows linearly with the n_sessions, and it gets slow pretty quick.
The output vector _weighted_sum has the length of the number of sessions, and just allocating it gets slow.
We can expect the number of sessions to grow in the millions, while only a small fraction of those are in scope
for one request.
So, next step is to get some kind of mapping to a smaller subset of session.

"""

n_items = 20000
n_sessions = 50000
current_session_len = 10
sessions_per_h_item = 1000

xp = cp
item_to_sessions = xp.random.randint(0, n_sessions, (n_items, sessions_per_h_item), dtype=cp.intc)
example_session = xp.random.randint(0, n_items, current_session_len)

# kernel; out array = n_sessions rows and items_in_session cols.
current_session = np.random.randint(0, n_items, current_session_len)
weights = (cp.arange(current_session_len, dtype=cp.float32) + 1) / current_session_len


# Question: Does this increase memory footprint or is it just a pointer ?
# Question: Can we avoid a copy ?
relevant_slice = item_to_sessions[current_session, :]

n_blocks_x = int(relevant_slice.shape[1] / 16) + 1
n_blocks_y = int(relevant_slice.shape[0] / 16) + 1

# todo: tests if faster when inserting & skipping zeroes
weighted_count_kernel = cp.RawKernel(r'''
extern "C" __global__
void weighted_count_kernel(const int* sessions, const float* weight, const int* unique_sessions,  
                            const int n_unique_sessions, 
                            const int n_items, const int n_sessions,  float* y) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int s_id = blockDim.z * blockIdx.z + threadIdx.z;

    
    if (row < n_items && col < n_sessions && s_id < n_unique_sessions){
        int session_id = sessions[(row * n_sessions + col)];
        if (session_id == unique_sessions[s_id]){
            //printf("row: %d col: %d session: %d gets added: %f \n", row, col, session_id, weight[row]);
            atomicAdd(&y[s_id], weight[row]); //&y[s_id], weight[row]
        }
   }
   else {
        //printf("row: %d > %d OR  col: %d  > %d  OR s_id: %d > %d \n", row, n_items, col, n_sessions, s_id, n_unique_sessions);
   }
}
''', 'weighted_count_kernel')

unique_sessions = cp.unique(relevant_slice)
n_unique_sessions = len(unique_sessions)

n_blocks_z = int(n_unique_sessions / 4) + 1

weighted_sum = cp.zeros((10,), dtype=np.float32)
weighted_count_kernel((n_blocks_x, n_blocks_y, n_blocks_z),
                      (16, 16, 4),
                      (relevant_slice, weights, unique_sessions, n_unique_sessions, current_session_len,
                       sessions_per_h_item, weighted_sum))
# print(weighted_sum)


def prep_stuff(_relevant_slice):
    unique_sessions = cp.unique(_relevant_slice)
    # n_unique_sessions = len(unique_sessions)
    # n_blocks_z = int(n_unique_sessions / 4) + 1
    # _weighted_sum = cp.zeros((n_unique_sessions,), dtype=np.float32)


print(repeat(prep_stuff, (relevant_slice,), n_repeat=1000))


def num_count(_relevant_slice, _weights, _current_session_len, _items_per_h_sessions):
    unique_sessions = cp.unique(_relevant_slice)
    n_unique_sessions = len(unique_sessions)
    n_blocks_z = int(n_unique_sessions / 4) + 1
    _weighted_sum = cp.zeros((n_unique_sessions,), dtype=np.float32)
    weighted_count_kernel(
        (n_blocks_x, n_blocks_y, n_blocks_z),
        (16, 16, 4),
        (_relevant_slice, _weights, unique_sessions, n_unique_sessions, _current_session_len, _items_per_h_sessions, _weighted_sum))

print(repeat(num_count, (relevant_slice, weights, current_session_len, sessions_per_h_item), n_repeat=10))


# def unique_values(_matrix_slice):
#     return cp.unique(_matrix_slice)
#
# n_sessions = 10 ** 6
# for mat_h, mat_w in [(10, 500), (50, 5000), (100, 10000)]:
#     random_mat = cp.random.randint(0, n_sessions, (mat_h, mat_w))
#     print(repeat(unique_values, (random_mat,), n_repeat=100))



"""
Conclusion:
Kerel much slower in 3D
Now time taken grows with sessions_per_h_item
3379.424 us for sessions_per_h_item == 5000
"""
