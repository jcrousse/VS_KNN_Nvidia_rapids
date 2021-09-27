# This is a one-off process to create a train and a test split of sessions.
# We don't bother optimizing it, just basic pandas / sklearn is enough
import json
import random
import pandas as pd


def train_test_split(test_size=5000):
    with open('config.json', 'r') as f:
        project_config = json.load(f)

    full_df = pd.read_csv(project_config['data_sources']['full_data'], names=['session', '_t', 'items', '_c', ])

    all_sessions = list(set(full_df['session'].values))

    n_sessions = len(all_sessions)
    print(f"Found {n_sessions} unique sessions in the dataset")

    random.seed(555)
    test_sessions = random.sample(all_sessions, test_size)

    train_data = full_df[~full_df['session'].isin(test_sessions)]
    test_data = full_df[full_df['session'].isin(test_sessions)]

    print(f"Train data has {len(train_data)} rows for {n_sessions - test_size} sessions")
    print(f"Test data has {len(test_data)} rows for {test_size} sessions")

    train_data.to_csv(project_config['data_sources']['train_data'], index=False, header=False)
    test_data.to_csv(project_config['data_sources']['test_data'], index=False, header=False)
