import cupy as cp
import json
from vs_knn.weighted_word_count import weighted_word_count
from vs_knn.index_builder import IndexBuilder
from cupyx.time import repeat


def vs_knn_predict(session, weights):
    item_slice = ITEM_TO_SESSIONS[session]
    unique_sessions, w_sum_sessions = weighted_word_count(item_slice, weights)
    if len(unique_sessions) > 0:  # todo: some items have no historical sessions? 31, 287
        if len(unique_sessions) > K_SESSIONS:
            # sort and select top K
            print("todo: implement sort & select here")
        session_slice = SESSION_TO_ITEMS[unique_sessions]
        unique_items, w_sum_items = weighted_word_count(session_slice, w_sum_sessions)
        return unique_items, w_sum_items


def run_random_test():
    random_session_id = cp.random.randint(0, MAX_SESSIONS)
    session = SESSION_TO_ITEMS[random_session_id]
    session_clean = session[cp.where(session > -1)]
    n_items = len(session_clean)
    items, scores = vs_knn_predict(session_clean, ITEM_WEIGHTS[-n_items:])
    return items, scores


with open('config.json', 'r') as f:
    project_config = json.load(f)


index_builder = IndexBuilder(project_config)
MAX_SESSIONS = 1000  # saving memory
MAX_SESSION_LEN = 100
K_SESSIONS = 100
index_builder.create_indices('train_data', save=False, max_sessions=MAX_SESSIONS)

SESSION_TO_ITEMS = index_builder.get_index_as_table('session')
ITEM_TO_SESSIONS = index_builder.get_index_as_table('item')

ITEMS_PER_SESSION = SESSION_TO_ITEMS.shape[1]
SESSIONS_PER_ITEM = ITEM_TO_SESSIONS.shape[1]
ITEM_WEIGHTS = (cp.arange(MAX_SESSION_LEN, dtype=cp.float32) + 1) / MAX_SESSION_LEN

if __name__ == '__main__':
    print(repeat(run_random_test, n_repeat=1000))
