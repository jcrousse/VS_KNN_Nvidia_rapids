import cupy as cp

copy_values_kernel = cp.RawKernel(r'''
extern "C" __global__
void copy_values_kernel(
                            const int* idx_array, 
                            const int* in_values, 
                            const int num_idx,
                            const int row_len,
                            const int col_len, 
                            const int buffer_len,
                            int* out_values) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int row = tid / row_len;
    int col = tid % row_len;

    if (tid < buffer_len){
        if (row < num_idx){
            int value_idx = idx_array[row * 2] + col;

            if (value_idx <= idx_array[row * 2 + 1]){
                out_values[tid] = in_values[value_idx];
                //printf("Reading value %d at row %d col %d and writing it at position %d \n", in_values[value_idx], row, col, tid);
            }
            else{
                out_values[tid] = 0;
            }
        }
        else{
            out_values[tid] = 0;
        }
    }

}
''', 'copy_values_kernel')

copy_weights_kernel = cp.RawKernel(r'''
extern "C" __global__
void copy_weights_kernel(
                            const int* idx_array, 
                            const float* in_weight,
                            const int num_idx,
                            const int row_len,
                            const int col_len, 
                            const int buffer_len,
                            float* out_weights) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int row = tid / row_len;
    int col = tid % row_len;
    
    if (tid < buffer_len){
        if (row < num_idx){
            int value_idx = idx_array[row * 2] + col;
            if (value_idx <= idx_array[row * 2 + 1]){
                //printf("Assigning weight %f at idx %d for row %d col %d\n", in_weight[row], tid, row, col);
                out_weights[tid] = in_weight[row];
            }
            else{
                out_weights[tid] = 0.0;
            }
        }
        else{
            out_weights[tid] = 0.0;
        }
    }

}
''', 'copy_weights_kernel')



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

