import cupy as cp

from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
import time


class SimpleModel:
    def __init__(self):
        pass

    def predict(self, N, power, use_cuda_streams=True):

        if use_cuda_streams:
            compute_stream = cp.cuda.stream.Stream(non_blocking=True)
            compute_stream.use()
        d_mat = cp.random.randn(N * N, dtype=cp.float64).reshape(N, N)
        d_ret = d_mat

        cp.matmul(d_ret, d_mat)

        for i in range(power - 1):
            d_ret = cp.matmul(d_ret, d_mat)

        if use_cuda_streams:
            compute_stream.synchronize()

        return 1


model = SimpleModel()


def multi_thread_calls(use_cuda_streams=True, n_calls=20):

    inputs = [{'N': 1024, 'power': 128}] * n_calls

    start = time.time()
    with ThreadPoolExecutor(10) as executor:
        futures = []
        for in_params in inputs:
            futures.append(executor.submit(model.predict, in_params['N'], in_params['power'], use_cuda_streams))
        for future in as_completed(futures):
            pass

    print(f"ran {n_calls} calls in {time.time() - start} seconds")


if __name__ == '__main__':
    multi_thread_calls(True)
    multi_thread_calls(False)
