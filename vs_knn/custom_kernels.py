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
    
    int row = tidx / row_len;
    int col = tidx % row_len;
    
    //int batch_id = tid / (n_unique_values * buffer_len);
    //int weight_read_idx = row % col_len + batch_id * ;

    if (tidx < (batch_len * batch_size)){
        out_values[tidx] = 0;
        out_weights[tidx] = 0.0;
        if (row < num_idx){
            int value_idx = idx_array[row * 2] + col;
            if (value_idx <= idx_array[row * 2 + 1] && value_idx > 0){
                out_values[tidx] = in_values[value_idx];
                out_weights[tidx] = in_weight[row];
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
                            const int num_batches,
                            float* out_weights) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int row = tid / n_unique_values;
    int col = tid % n_unique_values;
    int batch_id = tid / (n_unique_values * buffer_len);
    
    int in_value = in_values[row];   
    if (tid < (buffer_len * n_unique_values * num_batches) 
        && in_value == unique_values[col + n_unique_values * batch_id] 
        && in_value != 0){
        atomicAdd(&out_weights[col + n_unique_values * batch_id], in_weight[row]);
    }


}
''', 'groubpy_kernel')

