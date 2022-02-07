import cupy as cp
import numpy as np

# currently only works with 'int' types, but could be extended to any type
weighted_wordcount_kernel = cp.RawKernel(r'''
extern "C" __global__
void weighted_wordcount_kernel(
                            const int* word_matrix, 
                            const float* weight, 
                            const int* unique_words,  
                            const int num_rows, 
                            const int num_cols,  
                            const int n_unique_words, 
                            float* weighted_count) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int word_id = blockDim.z * blockIdx.z + threadIdx.z;

    if (row < num_rows && col < num_cols && word_id < n_unique_words){
        int word = word_matrix[(row * num_cols + col)];
        if (word != 0 && word == unique_words[word_id]){
            atomicAdd(&weighted_count[word_id], weight[row]);
        }
   }
}
''', 'weighted_wordcount_kernel')


def weighted_word_count(word_matrix, row_weights):
    """
    :param word_matrix: A 2-D CuPy array of words (integer 1...N). O means missing value and will be ignored
    :param row_weights: One weight value per row in the matrix. Words found in row i count for a weight row_weights|i]
    :return: Two vectors: The unique words found in the word matrix, and the weighted count of each word
    """
    if word_matrix.ndim == 1:
        word_matrix = cp.reshape(word_matrix, (1, word_matrix.shape[0]))
    elif word_matrix.ndim != 2:
        raise ValueError("word_matrix must be a 1-D or 2-D array")
    num_rows, num_cols = word_matrix.shape
    n_blocks_x = int(num_rows / 8) + 1
    n_blocks_y = int(num_cols / 8) + 1
    unique_words = cp.unique(word_matrix)
    if unique_words[0] == 0:
        unique_words = unique_words[1:]
    n_unique_words = len(unique_words)
    n_blocks_z = int(n_unique_words / 4) + 1
    weighted_count = cp.zeros((n_unique_words,), dtype=np.float32)
    weighted_wordcount_kernel(
        (n_blocks_x, n_blocks_y, n_blocks_z),
        (8, 8, 4),
        (word_matrix, row_weights, unique_words, num_rows, num_cols, n_unique_words, weighted_count)
    )
    # TODO: The number of threads per block should be a round multiple of the warp size,
    #  which is 32 on all current hardware.
    return unique_words, weighted_count
