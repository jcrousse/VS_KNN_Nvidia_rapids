import cupy as cp
import pickle
import tempfile
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed

tmp_filepath = tempfile.mktemp()

shared_array = cp.random.randint(0, 2, 10 ** 6)

memptr = shared_array.data

# fail at line https://github.com/cupy/cupy/blob/c74b5e632a5d0da983ea50d77d883d7b66d5001e/cupy/cuda/memory.pyx#L287

data_pointer = {
    "ptr": memptr.ptr,
    "shape": shared_array.shape,
    "dtype": shared_array.dtype
}

with open(tmp_filepath, 'wb') as f:
    pickle.dump(data_pointer, f)

with open(tmp_filepath, 'rb') as f:
    data_pointer_loaded = pickle.load(f)

mempool = cp.get_default_memory_pool()
basemem = cp.cuda.memory.BaseMemory(mempool)
basemem.ptr = data_pointer_loaded["ptr"]
ptr = cp.cuda.memory.MemoryPointer(basemem, 0)
# new_array = cp.ndarray(shape=shared_array.shape, memptr=memptr, dtype=shared_array.dtype)
new_array = cp.ndarray(shape=data_pointer_loaded['shape'], memptr=ptr, dtype=data_pointer_loaded['dtype'])

new_array[0] = 0
assert shared_array[0] == 0

del shared_array

slow_kernel = cp.RawKernel(r'''
extern "C" __global__
void slow_kernel(const int* shared_data, int* out) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset  < 100000)
    {
        clock_offset = clock() - start_clock;
    }
    out[0] = clock_offset ;
}
''', 'slow_kernel')
n_threads = len(shared_array)
t_per_block = 256
n_blocks = n_threads // t_per_block + 1
s1 = cp.cuda.Stream()
s2 = cp.cuda.Stream()

with s1:
    out_array = cp.zeros(1, dtype=cp.int)
    slow_kernel(
        (n_blocks,),
        (t_per_block,),
        (shared_array, out_array),
    )
    a = 1

with s2:
    out_array = cp.zeros(1, dtype=cp.int)
    slow_kernel(
        (n_blocks,),
        (t_per_block,),
        (shared_array, out_array),
    )


import time


class SharedDataModel:
    def __init__(self):
        # self.stream = cp.cuda.Stream(non_blocking=True)
        # self.stream.use()
        self.buffer = cp.array([])

        mempool = cp.get_default_memory_pool()
        basemem = cp.cuda.memory.BaseMemory(mempool)
        basemem.ptr = data_pointer_loaded["ptr"]
        ptr = cp.cuda.memory.MemoryPointer(basemem, 0)
        self.shared_array = \
            cp.ndarray(shape=data_pointer_loaded['shape'], memptr=ptr, dtype=data_pointer_loaded['dtype'])

    def predict(self):
        stream = cp.cuda.Stream(non_blocking=True)
        stream.use()  # needed?
        res = int(self.shared_array.sum())
        return res

    def __call__(self, *args, **kwargs):
        return self.predict()


class SimpleModel:
    def __init__(self):
        pass

    def predict(self, N, power):
        # compute_stream = cp.cuda.stream.Stream(non_blocking=True)
        # compute_stream.use()
        d_mat = cp.random.randn(N * N, dtype=cp.float64).reshape(N, N)
        d_ret = d_mat

        cp.matmul(d_ret, d_mat)

        for i in range(power - 1):
            d_ret = cp.matmul(d_ret, d_mat)
        # compute_stream.synchronize()

        return 1


model = SimpleModel()

if __name__ == '__main__':
    inputs = [{'N': 1024, 'power': 128}] * 100
    with ThreadPoolExecutor(3) as executor:
        futures = []
        for in_params in inputs:
            futures.append(executor.submit(model.predict, in_params['N'], in_params['power']))
        for future in as_completed(futures):
            print(future.result())




a = 1
