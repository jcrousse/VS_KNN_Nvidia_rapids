import cudf
import pandas as pd
from vs_knn.col_names import project_config

xd = cudf  # out of memory with cudf for the 'drop' operation on my GPU ?? using pandas instead


def reset_values(df, column):
    unique_values = df[column].unique()
    unique_values_map = xd.DataFrame(unique_values).reset_index()
    unique_values_map.columns = ['_new_val', column]
    new_values_df = df.merge(unique_values_map, on=column)
    new_values_df = new_values_df.drop(columns=[column])
    new_values_df = new_values_df.rename(columns={'_new_val': column})
    return new_values_df


def preprocess_data():
    full_df = xd.read_csv(project_config['data_sources']['raw_data'], names=['session', '_t', 'items', '_c', ])

    reset_sessions = reset_values(full_df, 'session')
    reset_sessions_items = reset_values(reset_sessions, 'items')

    preprocessed_df = reset_sessions_items[['session', '_t', 'items', '_c', ]]
    preprocessed_df = preprocessed_df.sort_values(by=['session', '_t'])
    preprocessed_df.to_csv(project_config['data_sources']['prep_data'], index=False, header=False)
