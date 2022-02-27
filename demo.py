import cudf
import os
import cupy as cp
from vs_knn.vs_knn import CupyVsKnnModel
import gc
import time
from tqdm import tqdm


dataset_filepath = 'archive/yoochoose-clicks.dat'


yoochoose_data = cudf.read_csv(dataset_filepath,
                               usecols=[0, 1, 2],
                               dtype={
                                 'session_id': cp.dtype('int32'),
                                 'item_id': cp.dtype('int32'),
                                 'timestamp': cp.dtype('O')
                                 },
                               names=['session_id', 'timestamp', 'item_id'])


n_rows = yoochoose_data.shape[0]
n_sessions = len(yoochoose_data['session_id'].unique())
n_items = len(yoochoose_data['item_id'].unique())
filesize = os.path.getsize(dataset_filepath)

print(f"the dataset contains {round(n_rows / 10 ** 6)}M rows, ",
      f"with {round(n_sessions / 10 ** 6)}M ",
      f"sessions and {round(n_items / 10 ** 3)}K items",
      f"\nOriginal file size: {round(filesize / 10 ** 6)}Mb")


yoochoose_data['day'] = yoochoose_data['timestamp'].str.slice(start=0, stop=10)

all_days = yoochoose_data['day'].unique()
train_days = all_days[0:180]
print(all_days)

train_df = yoochoose_data[yoochoose_data['day'].isin(train_days)][['session_id', 'timestamp', 'item_id']]
test_df = yoochoose_data[~yoochoose_data['day'].isin(train_days)][['session_id', 'timestamp', 'item_id']]

del yoochoose_data
gc.collect()


model = CupyVsKnnModel(top_k=100, max_sessions_per_items=5000, max_item_per_session=10)

start = time.time()
model.train(train_df)
end = time.time()
print(f"trained the model in {end - start} seconds")


def get_test_examples(test_set):
    test_array = test_set \
        .drop('timestamp', axis=1) \
        .groupby('session_id') \
        .agg({'item_id': 'collect'})['item_id']\
        .to_pandas()\
        .values
    return test_array


test_sessions_array = get_test_examples(test_df)


def session_to_xy(items_in_session):
    return (items_in_session[0:-1], items_in_session[-1]) if len(items_in_session) > 1 else (None, None)


def test_a_model(model, test_data):
    total_hits = 0
    n_treated = 0
    hr20 = 0

    pbar = tqdm(test_data)

    for test_session in pbar:
        x, y = session_to_xy(test_session)
        if x is not None:
            items_pred, item_scores = model.predict(x)
            n_treated += 1
            if len(items_pred) > 0:
                selection = cp.flip(cp.argsort(item_scores)[-20:])
                items_rec = items_pred[selection]

                if y in items_rec:
                    total_hits += 1
                    hr20 = total_hits / n_treated
                    pbar.set_postfix({'HR@20': hr20})

    time_per_iter = pbar.format_dict['elapsed'] / pbar.format_dict['n']

    return time_per_iter, hr20


itertime_rd, hr_rd = test_a_model(model, test_sessions_array)