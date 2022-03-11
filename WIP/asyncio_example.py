import cupy as cp
import time
import asyncio
from vs_knn import CupyVsKnnModel


async def predict(N, power):
    compute_stream = cp.cuda.stream.Stream(non_blocking=True)
    compute_stream.use()
    d_mat = cp.random.randn(N * N, dtype=cp.float64).reshape(N, N)
    d_ret = d_mat

    cp.matmul(d_ret, d_mat)

    start = time.time()
    for i in range(power - 1):
        d_ret = cp.matmul(d_ret, d_mat)

    pre_synch = time.time()
    await asyncio.sleep(2)
    compute_stream.synchronize()
    cpu_time = pre_synch - start
    gpu_time = time.time() - pre_synch
    print(f"CPU time: {cpu_time}, GPU time: {gpu_time}")
    return cpu_time, gpu_time


async def main(n):
    cpu_time, gpu_time = await predict(1024, n)
    single_request_time = round(cpu_time + gpu_time, 1)

    start = time.time()
    _ = await asyncio.gather(predict(1024, n), predict(1024, n), predict(1024, n), predict(1024, n))
    total_time = round(time.time() - start, 1)
    gain = round(total_time / (single_request_time * 4) * 100)

    print(f"Treated one request of size {n} in {cpu_time + gpu_time}\n "
          f"Treated 4 requests of size {n} in {total_time} seconds, instead "
          f"of  {4 * single_request_time}, ({gain}% of sequential operations)")


if __name__ == "__main__":
    asyncio.run(main(32))
    asyncio.run(main(512))
