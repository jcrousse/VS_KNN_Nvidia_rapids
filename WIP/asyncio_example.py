import cupy as cp

# from concurrent.futures.thread import ThreadPoolExecutor
# from concurrent.futures import as_completed
import time
from vs_knn.vs_knn import CupyVsKnnModel
# import pickle
# import asyncio


def predict(N, power):
    compute_stream = cp.cuda.stream.Stream(non_blocking=True)
    compute_stream.use()
    d_mat = cp.random.randn(N * N, dtype=cp.float64).reshape(N, N)
    d_ret = d_mat

    cp.matmul(d_ret, d_mat)

    start = time.time()
    for i in range(power - 1):
        d_ret = cp.matmul(d_ret, d_mat)

    pre_synch = time.time()

    compute_stream.synchronize()

    return pre_synch - start, time.time() - pre_synch


class SimpleModel:
    def __init__(self):
        pass

    @staticmethod
    def predict(N, power):
        return predict(N, power)



sm_model = SimpleModel()

if __name__ == "__main__":

    pre_sync, sync = predict(1024, 128)
    print(f"CPU time: {pre_sync}")
    print(f"GPU time: {sync}")
