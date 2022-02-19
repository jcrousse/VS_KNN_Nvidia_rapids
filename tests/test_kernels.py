import cupy as cp
from cupyx.time import repeat

sessions_per_item, items_per_session = 5000, 10
start_idx = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
len_per_idx = [10, 5, 2, 3, 5, 2, 10]
start_end_idx = [[e, e + l] for e, l in zip(start_idx, len_per_idx)]

values_array = cp.arange(1000000, dtype=cp.intc)
idx_array = cp.array(start_end_idx, dtype=cp.intc)
weight_array = cp.arange(len(idx_array), dtype=cp.float32) / len(idx_array)
test_session = cp.array([1, 2, 3, 6], dtype=cp.intc)

buffer_shape = sessions_per_item * items_per_session
out_values = cp.random.randint(1, 100, buffer_shape, dtype=cp.intc)
out_weights = cp.random.random(buffer_shape, dtype=cp.float32)

copy_values_kernel = cp.RawKernel(r'''
extern "C" __global__
void copy_values_kernel(
                            const int* idx_array, 
                            const int* in_values, 
                            const float* weight,
                            const int row_len,
                            const int col_len, 
                            const int buffer_len,
                            int* out_values,
                            float* out_weights) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    int row = tid / row_len;
    int col = tid % row_len;
    
    int value_idx = idx_array[row * 2] + col;
    
    if (tid < buffer_len){
        if (value_idx < idx_array[row * 2 + 1]){
            out_values[tid] = in_values[value_idx];
        }
        else{
            out_values[tid] = 0;
        }
    }

}
''', 'copy_values_kernel')


def test_copy_kernel():
    run_kernel()
    _ = [value_array_check(i) for i in [0, 100, 1000, 5000, 5005]]


def run_kernel():
    kernel_args = (idx_array, values_array, weight_array,
                   sessions_per_item, items_per_session, buffer_shape,
                   out_values, out_weights)
    t_per_block = 256
    n_blocks = int(len(values_array) / t_per_block) + 1
    copy_values_kernel((n_blocks, ), (t_per_block,), kernel_args)


def value_array_check(out_idx):
    idx_in_input_session = out_idx // sessions_per_item
    idx_in_hist_sessions = out_idx % sessions_per_item
    value_idx = start_idx[idx_in_input_session] + idx_in_hist_sessions
    if idx_in_hist_sessions > len_per_idx[idx_in_input_session] - 1:
        expected_value = 0
    else:
        expected_value = int(values_array[value_idx])

    assert expected_value == int(out_values[out_idx])


def test_copy_kernel_speed():
    print(repeat(run_kernel, n_repeat=100))
