# This is a one-off process to create a train and a tests split of sessions.
# We don't bother optimizing it, just basic pandas / sklearn is enough
import json
import random
import pandas as pd


def train_test_split(config_file='config.json', split_method='days', n=180):
    with open(config_file, 'r') as f:
        project_config = json.load(f)

    full_df = pd.read_csv(project_config['data_sources']['prep_data'], names=['session', '_t', 'items', 'day', '_c', ])

    if split_method == 'days':
        train_data, test_data = day_split(full_df, n)
    else:
        train_data, test_data = random_test_split(full_df, n)

    train_data.to_csv(project_config['data_sources']['train_data'], index=False, header=False)
    test_data.to_csv(project_config['data_sources']['test_data'], index=False, header=False)

# todo: clean up this mess (too similar)
# todo: solution for tights coupling of data schema. Use central "read_youchoose" kind of thing


def random_test_split(df, test_size=5000) -> (pd.DataFrame, pd.DataFrame):
    """
    Create train : test set by randomly selecting `test_size` sessions as test set
    """

    all_sessions = list(df['session'].unique())

    n_sessions = len(all_sessions)
    print(f"Found {n_sessions} unique sessions in the dataset")

    random.seed(555)
    test_sessions = random.sample(all_sessions, test_size)

    train_data = df[~df['session'].isin(test_sessions)]
    test_data = df[df['session'].isin(test_sessions)]

    print(f"Train data has {len(train_data)} rows for {n_sessions - test_size} sessions")
    print(f"Test data has {len(test_data)} rows for {test_size} sessions")

    return train_data, test_data


def day_split(df, n_days_train=180):
    """
    split the dataset by selecting the number of days to keep in the training set
    """
    all_days = list(df['day'].unique())
    train_days = all_days[0:n_days_train]

    train_data = df[df['day'].isin(train_days)]
    test_data = df[~df['day'].isin(train_days)]

    print(f"Train data has {len(train_data)} rows for {n_days_train} days")
    print(f"Test data has {len(test_data)} rows for {len(all_days) - n_days_train} days")

    return train_data, test_data
