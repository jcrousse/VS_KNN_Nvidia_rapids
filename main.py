import argparse
import json
import numpy as np
import cupy as cp
from cupyx.time import repeat
from vs_knn import CupyVsKnnModel
from vs_knn.train_test_split import train_test_split
from vs_knn.preprocessing import preprocess_data
from vs_knn.index_builder import IndexBuilder
from vs_knn.data_read_write import read_dataset
from vs_knn.col_names import SESSION_ID
from tqdm import tqdm

MAX_SESSIONS = None  # use value such as 10 ** 6 to save memory and time for development


def get_arguments():
    """Get this script's command line arguments"""
    parser = argparse.ArgumentParser(description='cuDF implementation of VS-KNN')
    parser.add_argument('--train', '-t', dest='train', action='store_true', help="train the model")
    parser.set_defaults(train=False)
    parser.add_argument('--split', '-s', dest='split', action='store_true', help="split the dataset into train/tests")
    parser.set_defaults(split=False)
    parser.add_argument('--preprocess', '-r', dest='preprocess', action='store_true',
                        help="data preprocessing: reset session and item values to contiguous integers")
    parser.set_defaults(preprocess=False)
    parser.add_argument('--predict', '-p', dest='predict', action='store_true', help="predict on tests dataset")
    parser.set_defaults(predict=False)
    parser.add_argument('--no-cudf', '-c', dest='no_cudf', action='store_true', help="use pandas instead of cudf")
    parser.set_defaults(no_cudf=False)
    args = parser.parse_args()
    return args.train, args.split, args.preprocess, args.predict, args.no_cudf


def setup_vsknn_indices(project_config, train_df):
    """ returns two key-value stores for session index and item index.
    At the moment it is a simple pandas dataframe behind the scenes, but any object that returns
    CuPy arrays should do"""
    items_per_sessions, sessions_per_item = \
        project_config['items_per_session'], project_config['sessions_per_item']
    train_dataset_path = project_config['data_sources']['train_data']

    index_builder = IndexBuilder(items_per_sessions, sessions_per_item)
    index_builder.create_indices(train_df, max_sessions=MAX_SESSIONS)

    # our key-value store is simply a pandas DataFrame
    session_index = index_builder.get_df_index('session', 'pandas')
    item_index = index_builder.get_df_index('item', 'pandas')
    return session_index, item_index


if __name__ == '__main__':
    train, split, preprocess, predict, no_cudf = get_arguments()

    with open('config.json', 'r') as f:
        project_config = json.load(f)

    if preprocess:
        preprocess_data(project_config)
    if split:
        train_test_split(project_config)

    if predict:
        train_set = read_dataset('train_data', project_config, 'cudf')
        test_set = read_dataset('test_data', project_config, 'cudf')

        session_to_items, item_to_sessions = setup_vsknn_indices(project_config, train_set)
        model = CupyVsKnnModel(item_to_sessions, session_to_items, top_k=project_config['top_k'])

        train_sessions = train_set[SESSION_ID].unique().values
        test_sessions = test_set[SESSION_ID].unique().values

        def run_random_test():
            """ function to run a prediction on a randomly selected session from train dataset for
            quick speed test """
            random_id = np.random.randint(0, len(train_sessions))
            random_session_id = train_sessions[random_id]
            session = session_to_items[random_session_id]
            session_clean = session[cp.where(session > 0)]
            items, scores = model.predict(session_clean)
            return items, scores

        print(repeat(run_random_test, n_repeat=100))

        session_to_items_test = test_set.set_index(SESSION_ID)

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
