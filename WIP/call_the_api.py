# import requests
# import time
import asyncio
from aiohttp import ClientSession
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
import requests
import pickle
import time

# TODO NEXT:
#  - Compare ins/outs 1 by 1 between demo and this API call. Why do we get shit results?
#  - Increase number of threads, does it break stuff? Overwriting on same data?
#  - Measure the speed
#  - Can we then speedup by using a nex stream for every predict?

def request_post(x, y):
    r = requests.post("http://127.0.0.1:8000", params={"q": x})
    rec_items = r.json()['recommended_items']
    return 1 if y in rec_items else 0


async def fetch_html(url: str, session: ClientSession, **kwargs) -> str:
    """GET request wrapper to fetch page HTML.

    kwargs are passed to `session.request()`.
    """

    resp = await session.request(method="GET", url=url, **kwargs)
    html = await resp.text()
    return html


async def bulk_crawl_and_write(urls: list) -> None:
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(
                fetch_html(url=url, session=session)
            )
        await asyncio.gather(*tasks)


def session_to_xy(items_in_session):
    return (items_in_session[0:-1], items_in_session[-1]) if len(items_in_session) > 1 else (None, None)


if __name__ == '__main__':

    with open('../test_data.pkl', 'rb') as f:
        test_array = pickle.load(f)

    x, y = [], []
    for test_sess in test_array[:2000]:
        _x, _y = session_to_xy(test_sess)
        if _x is not None:
            x.append(_x)
            y.append(_y)

    total_hits = 0
    n_queries = len(x)

    start = time.time()

    with ThreadPoolExecutor(3) as executor:
        futures = []
        for t_sess, target in zip(x, y):
            futures.append(executor.submit(request_post, t_sess, target))
        for future in as_completed(futures):
            total_hits += future.result()

    time_taken = time.time() - start
    qps = n_queries / time_taken

    print(f"treated {n_queries} in {round(time_taken, 2)} seconds ({round(qps)} per second). \n"
          f"HR@20: {total_hits / n_queries}")


    #
    # asyncio.run(bulk_crawl_and_write(urls=urls))




