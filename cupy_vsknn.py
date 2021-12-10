import cupy as cp
import cudf
import json
from vs_knn import CupyVsKnnModel
from vs_knn.index_builder import IndexBuilder
from cupyx.time import repeat
from tqdm import tqdm
import numpy as np

# TODO:
#  - Set selection of Train / test based on date. e.g. 180 days of train
#  - Remove duplicated sessions from train
#  - Try python hashmap to CuPy arrays as index ?

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        project_config = json.load(f)

    index_builder = IndexBuilder(project_config)
    MAX_SESSIONS = None  # use value such as 10 ** 6 to save memory
    MAX_SESSION_LEN = 100
    K_SESSIONS = 100

    index_builder.create_indices('train_data', save=False, max_sessions=MAX_SESSIONS)


    # SESSION_TO_ITEMS = index_builder.get_index_as_array('session')
    # ITEM_TO_SESSIONS = index_builder.get_index_as_array('item')
    # SESSION_TO_ITEMS = index_builder.get_dict_index('session')
    # ITEM_TO_SESSIONS = index_builder.get_dict_index('item')
    SESSION_TO_ITEMS = index_builder.get_df_index('session', 'pandas')
    ITEM_TO_SESSIONS = index_builder.get_df_index('item', 'pandas')

    ITEMS_PER_SESSION = SESSION_TO_ITEMS.shape[1]
    SESSIONS_PER_ITEM = ITEM_TO_SESSIONS.shape[1]

    model = CupyVsKnnModel(project_config, ITEM_TO_SESSIONS, SESSION_TO_ITEMS)
    sessions_train = cp.asnumpy(index_builder.get_unique_sessions())

    def run_random_test():
        random_id = np.random.randint(0, len(sessions_train))
        random_session_id = sessions_train[random_id]
        session = SESSION_TO_ITEMS[random_session_id]
        session_clean = session[cp.where(session > 0)]
        items, scores = model.predict(session_clean)
        return items, scores

    print(repeat(run_random_test, n_repeat=100))

    test_set = cudf.read_csv(project_config['data_sources']['test_data'],
                             names=['session_id', 'time', 'item_id'],
                             dtype={
                                 'session_id': cp.dtype('int32'),
                                 'time': cp.dtype('O'),
                                 'item_id': cp.dtype('int32')
                             },
                             usecols=[0, 1, 2]
                             )
    test_sessions = test_set['session_id'].unique().values
    session_to_items_test = test_set.set_index("session_id")

    total_hits = 0
    pbar = tqdm(test_sessions)
    for n_treated, test_session in enumerate(pbar):
        test_items = session_to_items_test.loc[test_session]['item_id'].values
        if len(test_items) > 1:
            x = test_items[0:-1]
            y = test_items[-1]
            items_pred, item_scores = model.predict(x)

            selection = cp.flip(cp.argsort(item_scores)[-20:])
            items_rec = items_pred[selection]

            if y in items_rec:
                total_hits += 1
            pbar.set_postfix({'HR@20': total_hits / n_treated})

    print(f"HR@20: {total_hits / len(test_sessions)}")


