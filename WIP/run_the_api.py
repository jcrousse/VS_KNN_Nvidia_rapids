from fastapi import FastAPI, Query
from typing import List
from vs_knn.vs_knn import CupyVsKnnModel
import uvicorn
import cupy as cp

# for minibatch API check this out:
# https://levelup.gitconnected.com/fastapi-how-to-process-incoming-requests-in-batches-b384a1406ec

app = FastAPI()

model = CupyVsKnnModel()
model.load('../saved_model')

@app.post("/")
def root(q: List[int] = Query(None)):
    prediction = model.predict(q)
    items_pred, item_scores = prediction['predicted_items'], prediction['scores']
    items_rec = []
    if len(items_pred) > 0:
        selection = cp.flip(cp.argsort(item_scores)[-20:])
        items_rec = cp.asnumpy(items_pred[selection]).tolist()

    return {"recommended_items": items_rec}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

