# import requests
# import time
import asyncio
from aiohttp import ClientSession
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
import requests
import pickle


def request_post(x, y):
    r = requests.post("http://127.0.0.1:8000", params={"q": x})
    rec_items = r.json()['recommended_items']
    if y in rec_items:
        res = f"item {y} found in recommendations"
    else:
        res = f"item {y} NOT found in recommendations"
    return res


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
    for test_sess in test_array:
        _x, _y = session_to_xy(test_sess)
        if _x is not None:
            x.append(_x)
            y.append(_y)

    with ThreadPoolExecutor(1) as executor:
        futures = []
        for t_sess, target in zip(x, y):
            futures.append(executor.submit(request_post, t_sess, target))
        for future in as_completed(futures):
            print(future.result())

    #
    # asyncio.run(bulk_crawl_and_write(urls=urls))




