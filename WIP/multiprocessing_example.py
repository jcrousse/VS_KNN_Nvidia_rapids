import time
from fastapi import FastAPI, Query
from typing import List
from concurrent.futures import ProcessPoolExecutor
import asyncio
import uvicorn
import os

app = FastAPI()


def predict(q):
    time.sleep(0.002)
    pid = os.getpid()
    return {"recommended_items": [1, 2, 3]}


@app.post("/")
async def root(q: List[int] = Query(None)):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(2) as pool:
        result = await loop.run_in_executor(pool, predict, q)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
