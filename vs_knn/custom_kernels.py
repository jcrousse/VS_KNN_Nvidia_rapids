import cupy as cp

copy_values_kernel = cp.RawKernel(r'''
extern "C" __global__
void copy_values_kernel(
                            const int* idx_array, 
                            const int* in_values, 
                            const float* in_weight,
                            const int num_idx,
                            const int row_len,
                            const int col_len, 
                            const int batch_len,
                            const int batch_size,
                            int* out_values,
                            float* out_weights) {

    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int batch_id = tidx / batch_len;
    
    int row = tidx / row_len;
    int col = tidx % row_len;
    
    int write_idx = (tidx - batch_len * batch_id) * batch_size + batch_id;

    if (tidx < (batch_len * batch_size)){
        out_values[write_idx] = 0;
        out_weights[write_idx] = 0.0;
        if (row < num_idx){
            int value_idx = idx_array[row * 2] + col;
            if (value_idx <= idx_array[row * 2 + 1]){
                out_values[write_idx] = in_values[value_idx];
                out_weights[write_idx] = in_weight[row];
                //printf("Reading value %d at row %d col %d and writing it at position %d \n", in_values[value_idx], row, col, tid);
            }
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
    }


}
''', 'groubpy_kernel')

