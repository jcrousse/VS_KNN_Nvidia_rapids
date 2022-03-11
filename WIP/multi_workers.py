import gc
import time
from fastapi import FastAPI, Query
from typing import List
import uvicorn
import cupy as cp
from vs_knn import CupyVsKnnModel

app = FastAPI()

model = CupyVsKnnModel()
model.load('../saved_model')


def predict(q):
    prediction = model.predict(q)
    items_pred, item_scores = prediction['predicted_items'], prediction['scores']
    selection = cp.flip(cp.argsort(item_scores)[-20:])
    items_rec = cp.asnumpy(items_pred[selection]).tolist()
    return {"recommended_items": items_rec}


@app.post("/")
def root(q: List[int] = Query(None)):
    result = predict(q)
    return result


if __name__ == "__main__":
    uvicorn.run("multi_workers:app", host="0.0.0.0", port=8000, workers=4)
