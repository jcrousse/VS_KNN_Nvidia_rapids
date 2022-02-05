import cudf
import random
from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID
from vs_knn.vs_knn import CupyVsKnnModel
from cupyx.time import repeat


def predict_vsknn(vsknn_model, list_of_sessions):
    session = list_of_sessions[random.randint(0, 1000)]
    return vsknn_model.predict(session)


if __name__ == '__main__':

    train_data_path = "archive/yoochoose-clicks.dat"
    train_df = cudf.read_csv(train_data_path,
                             usecols=[0, 1, 2],
                             names=[SESSION_ID, TIMESTAMP, ITEM_ID],
                             )

    model = CupyVsKnnModel(top_k=100, max_sessions_per_items=5000)
    model.train(train_df)

    random.seed(674837438)
    sessions = random.choices(train_df[SESSION_ID].unique(), k=1000)
    session_items = [
        [int(e) for e in train_df[train_df[SESSION_ID] == session][ITEM_ID].values]
        for session in sessions
    ]

    print(repeat(predict_vsknn, (model, session_items), n_repeat=10))
