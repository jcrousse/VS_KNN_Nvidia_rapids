import gc
import time
import os
import argparse
import cudf
import cupy as cp
from vs_knn.vs_knn import CupyVsKnnModel
from tqdm import tqdm
import pickle


def get_arguments():
    """Get this script's command line arguments"""
    parser = argparse.ArgumentParser(description='cuDF implementation of VS-KNN')
    parser.add_argument('--train', '-t', dest='train', action='store_true', help="train the model")
    parser.set_defaults(train=False)
    args = parser.parse_args()
    return args.train


def read_dataset(filepath, columns=None, delimiter=','):
    columns = ['session_id', 'timestamp', 'item_id'] if columns is None else columns
    return cudf.read_csv(filepath,
                         usecols=[0, 1, 2],
                         dtype={
                             'session_id': cp.dtype('int32'),
                             'item_id': cp.dtype('int32'),
                             'timestamp': cp.dtype('O')
                         },
                         delimiter=delimiter,
                         names=columns)


def train():
    dataset_filepath = 'archive/yoochoose-clicks.dat'

    yoochoose_data = read_dataset(dataset_filepath)

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

    trained_model = CupyVsKnnModel(top_k=100, max_sessions_per_items=5000, max_item_per_session=10)

    start = time.time()
    trained_model.train(train_df)
    end = time.time()
    print(f"trained the model in {end - start} seconds")

    test_sessions_array = get_test_examples(test_df)

    return trained_model, test_sessions_array


def train_session_rec_repo():
    train_filepath = "data/yoochoose-clicks-100k_train_full.txt"
    test_filepath = "data/yoochoose-clicks-100k_test.txt"
    delimiter = '\t'
    columns = ['session_id', 'item_id', 'timestamp']
    train_df = read_dataset(train_filepath, columns, delimiter)
    test_df = read_dataset(test_filepath, columns, delimiter)

    trained_model = CupyVsKnnModel(top_k=100, max_sessions_per_items=1000, max_item_per_session=10, decay='quadratic')
    trained_model.train(train_df)

    test_sessions_array = get_test_examples(test_df)

    return trained_model, test_sessions_array


def get_test_examples(test_set):
    test_array = test_set \
        .drop('timestamp', axis=1) \
        .groupby('session_id') \
        .agg({'item_id': 'collect'})['item_id']\
        .to_pandas()\
        .values
    return test_array


def session_to_xy(items_in_session):
    return (items_in_session[0:-1], items_in_session[-1]) if len(items_in_session) > 1 else (None, None)


def test_a_model(model, test_data):
    total_hits = 0
    n_treated = 0
    hr20 = 0

    pbar = tqdm(test_data)

    total_cpu = []
    total_gpu = []

    for test_session in pbar:
        x, y = session_to_xy(test_session)
        if x is not None:
            prediction = model.predict(x)
            items_pred, item_scores = prediction['predicted_items'], prediction['scores']
            total_cpu.append(prediction['cpu_time'])
            total_gpu.append(prediction['gpu_time'])
            n_treated += 1
            if len(items_pred) > 0:
                selection = cp.flip(cp.argsort(item_scores)[-20:])
                items_rec = items_pred[selection]

                if y in items_rec:
                    total_hits += 1
                    hr20 = total_hits / n_treated
                    pbar.set_postfix({'HR@20': hr20})

    time_per_iter = pbar.format_dict['elapsed'] / pbar.format_dict['n']

    print(f"total cpu time: {sum(total_cpu)}, total GPU time: {sum(total_gpu)}")

    return time_per_iter, hr20


if __name__ == '__main__':
    train_flag = get_arguments()

    if train_flag:
        model, test_array = train()
        model.save('saved_model')
        with open('test_data.pkl', 'wb') as f:
            pickle.dump(test_array, f)

    else:
        model = CupyVsKnnModel()
        model.load('saved_model')
        with open('test_data.pkl', 'rb') as f:
            test_array = pickle.load(f)

    itertime_rd, hr_rd = test_a_model(model, test_array[:1000])
    print(f"Ran predictions on {len(test_array)} test examples in {itertime_rd} seconds. HR@20: {hr_rd}")

    model2, test_array2 = train_session_rec_repo()
    itertime_rd, hr_rd = test_a_model(model2, test_array2)
    print(f"Ran predictions on {len(test_array2)} test examples in {itertime_rd} seconds. HR@20: {hr_rd}")
