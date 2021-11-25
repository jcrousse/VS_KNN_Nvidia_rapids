import cupy as cp
import json
from vs_knn import CupyVsKnnModel
from vs_knn.index_builder import IndexBuilder
from cupyx.time import repeat


if __name__ == '__main__':
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

    model = CupyVsKnnModel(project_config, ITEM_TO_SESSIONS, SESSION_TO_ITEMS, ITEM_WEIGHTS)

    def run_random_test():
        random_session_id = cp.random.randint(0, MAX_SESSIONS)
        session = SESSION_TO_ITEMS[random_session_id]
        session_clean = session[cp.where(session > -1)]
        items, scores = model.predict(session_clean)
        return items, scores


    print(repeat(run_random_test, n_repeat=1000))
