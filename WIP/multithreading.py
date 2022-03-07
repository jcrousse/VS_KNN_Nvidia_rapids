import cupy as cp

from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
import time
from vs_knn.vs_knn import CupyVsKnnModel
import pickle


class SimpleModel:
    def __init__(self):
        pass

    def predict(self, N, power, use_own_stream=True):

        if use_own_stream:
            compute_stream = cp.cuda.stream.Stream(non_blocking=True)
            compute_stream.use()
        d_mat = cp.random.randn(N * N, dtype=cp.float64).reshape(N, N)
        d_ret = d_mat

        cp.matmul(d_ret, d_mat)

        start = time.time()
        for i in range(power - 1):
            d_ret = cp.matmul(d_ret, d_mat)

        pre_synch = time.time()
        if use_own_stream:
            compute_stream.synchronize()

        return pre_synch - start, time.time() - pre_synch


sm_model = SimpleModel()
vsk_model = CupyVsKnnModel()
vsk_model.load('../saved_model')
# with open('../test_data.pkl', 'rb') as f:
#     test_array = pickle.load(f)


def multi_thread_calls(model, inputs, use_own_stream=True):

    presynch_time, synch_time, pct_time = [], [], []
    start = time.time()
    with ThreadPoolExecutor(1) as executor:
        futures = []
        for in_params in inputs:
            futures.append(executor.submit(model.predict, **in_params, use_own_stream=use_own_stream))
            # futures.append(executor.submit(model.predict, in_params['x'], use_cuda_streams))
        for future in as_completed(futures):
            pre_synch, synch = future._result
            presynch_time.append(pre_synch)
            synch_time.append(synch)
            gpu_pct = round((100 * synch) / (synch + pre_synch), 2)
            pct_time.append(gpu_pct)

    print(f"CPU time: {sum(presynch_time)}, GPU time {sum(synch_time)}, GPU% = {sum(pct_time) / len(pct_time)}")

    p_res = f"ran {len(inputs)} calls in {time.time() - start} seconds."
    p_res = p_res + " Using CUDA streams" if use_own_stream else p_res + "NOT using CUDA steams"
    print(p_res)


if __name__ == '__main__':
    sm_calls = 20
    vsk_calls = 1000
    sm_input = [{'N': 1024, 'power': 128}] * sm_calls
    vsk_input = [{'query_items': [214716935]}] * vsk_calls

    multi_thread_calls(sm_model, sm_input,  True)
    multi_thread_calls(sm_model, sm_input,  False)
    multi_thread_calls(vsk_model, vsk_input,  True)
    multi_thread_calls(vsk_model, vsk_input,  False)
