# This is a one-off process to create a train and a tests split of sessions.
# Not a lot of effort invested in optimizing this part as it is less critical
import random
import pandas as pd
from vs_knn.data_read_write import read_dataset, write_dataframe
from vs_knn.col_names import SESSION_ID, ITEM_ID, DAY


def train_test_split(project_config, split_method='days', n=180):

    full_df = read_dataset('prep_data', project_config, 'pandas')

    if split_method == 'days':
        train_data, test_data = day_split(full_df, n)
    else:
        train_data, test_data = random_test_split(full_df, n)

    train_data, test_data = remove_new_items_test(train_data, test_data)

    write_dataframe(train_data, 'train_data', project_config)
    write_dataframe(test_data, 'test_data', project_config)


def random_test_split(df, test_size=5000) -> (pd.DataFrame, pd.DataFrame):
    """
    Create train : test set by randomly selecting `test_size` sessions as test set
    """

    all_sessions = list(df[SESSION_ID].unique())

    n_sessions = len(all_sessions)
    print(f"Found {n_sessions} unique sessions in the dataset")

    random.seed(555)
    test_sessions = random.sample(all_sessions, test_size)

    train_data = df[~df[SESSION_ID].isin(test_sessions)]
    test_data = df[df[SESSION_ID].isin(test_sessions)]

    print(f"Train data has {len(train_data)} rows for {n_sessions - test_size} sessions")
    print(f"Test data has {len(test_data)} rows for {test_size} sessions")

    return train_data, test_data


def remove_new_items_test(train_data, test_data):
    """NOTE: this is the incorrect way to avoid issues with unknown items at inference,
    it will make the performance metric look artificially high"""
    items_train = set(train_data[ITEM_ID].unique())

    test_data = test_data[test_data[ITEM_ID].isin(items_train)]
    return train_data, test_data


def day_split(df, n_days_train=180):
    """
    split the dataset by selecting the number of days to keep in the training set
    """
    all_days = list(df[DAY].unique())
    train_days = all_days[0:n_days_train]

    train_data = df[df[DAY].isin(train_days)]
    test_data = df[~df[DAY].isin(train_days)]

    print(f"Train data has {len(train_data)} rows for {n_days_train} days")
    print(f"Test data has {len(test_data)} rows for {len(all_days) - n_days_train} days")

    return train_data, test_data
