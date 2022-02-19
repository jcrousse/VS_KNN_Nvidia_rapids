import cupy as cp
import cudf
from cupyx.time import repeat

sessions_per_item, items_per_session = 5000, 10
start_idx = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
len_per_idx = [10, 5, 2, 3, 5, 2, 10]
start_end_idx = [[e, e + l] for e, l in zip(start_idx, len_per_idx)]

values_array = cp.arange(1000000, dtype=cp.intc)
idx_array = cp.array(start_end_idx, dtype=cp.intc)
test_session = cp.array([1, 2, 3, 6], dtype=cp.intc)
weight_array = cp.arange(len(test_session), dtype=cp.float32) / len(test_session)

buffer_shape = sessions_per_item * items_per_session
out_values = cp.random.randint(1, 100, buffer_shape, dtype=cp.intc)
out_weights = cp.random.random(buffer_shape, dtype=cp.float32)

values_buffer = cp.random.randint(0, 500, buffer_shape, dtype=cp.intc)
weight_buffer = cp.random.random(buffer_shape, dtype=cp.float32)

unique_values = cp.unique(values_buffer)

copy_values_kernel = cp.RawKernel(r'''
extern "C" __global__
void copy_values_kernel(
                            const int* idx_array, 
                            const int* in_values, 
                            const float* in_weight,
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
            out_weights[tid] = in_weight[row];
        }
        else{
            out_values[tid] = 0;
            out_weights[tid] = 0;
        }
    }

}
''', 'copy_values_kernel')

groubpy_kernel = cp.RawKernel(r'''
extern "C" __global__
void groubpy_kernel(
                            const int* in_values, 
                            const float* in_weight,
                            const int* unique_values,
                            const int n_unique_values,
                            const int buffer_len,
                            float* out_weights) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int row = tid / n_unique_values;
    int col = tid % n_unique_values;

    int in_value = in_values[row];
    if (tid < buffer_len && in_values[row] == unique_values[col]){
        atomicAdd(&out_weights[col], in_weight[row]);
        //printf("row: %d col: %d value: %f \n", row, col, in_weight[row]);
    }


}
''', 'groubpy_kernel')


def test_copy_kernel():
    run_copy_kernel()
    _ = [value_array_check(i) for i in [0, 100, 1000, 5000, 5005]]


def run_copy_kernel():
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
        expected_weight = 0.0
    else:
        expected_value = int(values_array[value_idx])
        expected_weight = float(weight_array[idx_in_input_session])

    assert expected_value == int(out_values[out_idx])
    assert expected_weight == float(out_weights[out_idx])


def test_copy_kernel_speed():
    print(repeat(run_copy_kernel, n_repeat=1000))


def test_groupby_kernel_speed():
    print(repeat(run_groupby_kernel, n_repeat=1000))


def test_groupby_kernel():
    calc_weights = run_groupby_kernel()
    weight_array_check(calc_weights)


def run_groupby_kernel():
    out_weights_groupby = cp.zeros_like(unique_values, dtype=cp.float32)
    n_items = len(values_buffer) * len(unique_values)
    kernel_args = (values_buffer, weight_buffer, unique_values,
                   len(unique_values), n_items,
                   out_weights_groupby)
    t_per_block = 256
    n_blocks = int( n_items/ t_per_block) + 1
    groubpy_kernel((n_blocks, ), (t_per_block,), kernel_args)
    return out_weights_groupby


def weight_array_check(calculated_weights):
    df = cudf.DataFrame(data={
        'group_key': values_buffer,
        'agg_val': weight_buffer
    })
    df = df.groupby('group_key').agg({'agg_val': 'sum'}).sort_index()
    expected_weights = df['agg_val'].values

    def f_to_i(x):
        return round(float(x) * 100)
    comparison_key = [int(ek) == int(ck) for ek, ck in zip(unique_values, df.index.values)]
    comparison_weights = [f_to_i(ew) == f_to_i(cw) for ew, cw in zip(expected_weights, calculated_weights)]

    assert all(comparison_key) and all(comparison_weights)
