import cupy as cp
import numpy as np
from cupyx.time import repeat

"""
Fast, 
but the timz taken grows linearly with the n_sessions, and it gets slow pretty quick.
The output vector _weighted_sum has the length of the number of sessions, and just allocating it gets slow.
We can expect the number of sessions to grow in the millions, while only a small fraction of those are in scope
for one request.
So, next step is to get some kind of mapping to a smaller subset of session.

"""

n_items = 200000
n_sessions = 10000
current_session_len = 3
items_per_h_sessions = 500

xp = cp
item_to_sessions = xp.random.randint(0, n_sessions, (n_items, items_per_h_sessions), dtype=cp.intc)
example_session = xp.random.randint(0, n_items, current_session_len)

# kernel; out array = n_sessions rows and items_in_session cols.
current_session = np.random.randint(0, n_items, current_session_len)
weights = cp.arange(current_session_len, dtype=cp.float32) + 1 / current_session_len


# Question: Does this increase memory footprint or is it just a pointer ?
# Question: Can we avoid a copy ?
relevant_slice = item_to_sessions[current_session, :]

_bincount_kernel = cp.ElementwiseKernel(
    'T x', 'raw S bin',
    'atomicAdd(&bin[x], S(1))',
    'bincount_kernel')

_bincount_with_weight_kernel = cp.ElementwiseKernel(
    'S x, T w', 'raw U bin',
    'atomicAdd(&bin[x], w)',
    'bincount_with_weight_kernel')

n_blocks_x = int(n_sessions / 32) + 1
n_blocks_y = int(n_items / 32) + 1

weighted_count_kernel = cp.RawKernel(r'''
extern "C" __global__
void weighted_count_kernel(const int* sessions, const float* weight, const int n_items, const int n_sessions,  float* y) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < n_items && col < n_sessions)
    {
        int session_id = sessions[(row * n_sessions + col)];
        //printf("row: %d col: %d session: %d \n", row, col, session_id);
        atomicAdd(&y[session_id], weight[row]); //&y[session_id], weight[row]
    }
}
''', 'weighted_count_kernel')

weighted_sum = cp.zeros((10,), dtype=np.float32)
weighted_count_kernel((n_blocks_x, n_blocks_y),
                      (32, 32),
                      (relevant_slice, weights, current_session_len, items_per_h_sessions, weighted_sum))
# print(relevant_slice)

def num_count(_relevant_slice, _weights, _current_session_len, _items_per_h_sessions):
    _weighted_sum = cp.zeros((n_sessions,), dtype=np.float32)
    weighted_count_kernel(
        (n_blocks_x, n_blocks_y),
        (32, 32),
        (_relevant_slice, _weights, _current_session_len, _items_per_h_sessions, _weighted_sum))

print(repeat(num_count, (relevant_slice, weights, current_session_len, items_per_h_sessions), n_repeat=1000))
