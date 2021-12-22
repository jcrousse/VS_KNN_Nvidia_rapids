import pandas as pd
import cudf
from vs_knn.data_read_write import read_dataset, write_dataframe
from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID, DAY

xd = pd


def reset_values(df, column):
    tmp_col_name = '_new_val'
    unique_values = df[column].unique()
    unique_values_map = xd.DataFrame(unique_values).reset_index()
    unique_values_map.columns = [tmp_col_name, column]
    unique_values_map[tmp_col_name] = unique_values_map[tmp_col_name] + 1  # index starting at 1 so 0 is missing value
    new_values_df = df.merge(unique_values_map, on=column)
    new_values_df = new_values_df.drop(columns=[column])  # out of memory with cudf for the 'drop' operation on my GPU ?? using pandas instead
    new_values_df = new_values_df.rename(columns={tmp_col_name: column})
    return new_values_df


def preprocess_data(project_config):
    full_df = read_dataset('raw_data', project_config, 'pandas')

    reset_ids = project_config['reset_ids']
    if reset_ids:
        reset_sessions = reset_values(full_df, SESSION_ID)
        full_df = reset_values(reset_sessions, ITEM_ID)

    if DAY not in full_df.columns:
        full_df = get_unique_day_values(full_df)

    preprocessed_df = full_df.sort_values(by=[SESSION_ID, TIMESTAMP])

    write_dataframe(preprocessed_df, 'prep_data', project_config)


def get_unique_day_values(df):
    df[DAY] = df[TIMESTAMP].str.slice(start=0, stop=10)
    return df
