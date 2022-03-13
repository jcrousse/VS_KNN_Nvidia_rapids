from fastapi import FastAPI, Query
from typing import List
import uvicorn
import cupy as cp
from vs_knn import CupyVsKnnModel
import sys

app = FastAPI()


async def predict(q):
    prediction = await model.predict(q)
    items_pred, item_scores = prediction['predicted_items'], prediction['scores']
    selection = cp.flip(cp.argsort(item_scores)[-20:])
    items_rec = cp.asnumpy(items_pred[selection]).tolist()
    return {"recommended_items": items_rec}


@app.post("/")
async def root(q: List[int] = Query(None)):
    result = await predict(q)
    return result


if __name__ == "__main__":
    num_workers = sys.argv[0]
    model = CupyVsKnnModel()
    model.load('../saved_model')
    model.save_shared_pointers('model_ptr.pkl')
    uvicorn.run("multi_workers:app", host="0.0.0.0", port=8000, workers=num_workers)
else:
    model = CupyVsKnnModel()
    model.load('../saved_model')
    # model.load_shared_pointers('model_ptr.pkl')
