import cudf
import random

from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID
from vs_knn.vs_knn import CupyVsKnnModel
from vs_knn.weighted_word_count import weighted_word_count
from cupyx.time import repeat
import pickle
import os


def predict_vsknn(vsknn_model, list_of_sessions):
    session = list_of_sessions[random.randint(0, 999)]
    return vsknn_model.predict(session)


def get_test_sessions(df):
    stored_sessions_file = 'train_session.pkl'
    if not os.path.isfile(stored_sessions_file):
        random.seed(674837438)
        sessions = random.choices(df[SESSION_ID].unique(), k=1000)
        ret = [
            [int(e) for e in df[df[SESSION_ID] == session][ITEM_ID].values]
            for session in sessions
        ]
        with open(stored_sessions_file, 'wb') as f:
            pickle.dump(ret, f)
    else:
        with open(stored_sessions_file, 'rb') as f:
            ret = pickle.load(f)
    return ret


if __name__ == '__main__':
    import cupy as cp
    cp.random.seed(7357)
    test_matrix = cp.reshape(cp.arange(50000), (10, 5000))
    # Time proportional to n_unique_words
    # test_matrix = cp.random.randint(10, 500000, (6, 5000), dtype=cp.int32)
    weight_array_ones = cp.ones(10, dtype=cp.float32)
    print(repeat(weighted_word_count, (test_matrix, weight_array_ones), n_repeat=100))

    train_data_path = "archive/yoochoose-clicks.dat"
    train_df = cudf.read_csv(train_data_path,
                             usecols=[0, 1, 2],
                             names=[SESSION_ID, TIMESTAMP, ITEM_ID],
                             )

    model = CupyVsKnnModel(top_k=100, max_sessions_per_items=5000, max_item_per_session=10)
    model.train(train_df)
    print("model trained!")
    model.predict([214806142])

    session_items = get_test_sessions(train_df)
    print("Train exammples prepared")
    print(repeat(predict_vsknn, (model, session_items), n_repeat=1000))
