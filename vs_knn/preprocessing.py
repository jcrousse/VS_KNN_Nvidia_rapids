import cudf
import pandas as pd

xd = pd  # out of memory with cudf for the 'drop' operation on my GPU ?? using pandas instead


def reset_values(df, column):
    tmp_col_name = '_new_val'
    unique_values = df[column].unique()
    unique_values_map = xd.DataFrame(unique_values).reset_index()
    unique_values_map.columns = [tmp_col_name, column]
    unique_values_map[tmp_col_name] = unique_values_map[tmp_col_name] + 1  # index starting at 1 so 0 is missing value
    new_values_df = df.merge(unique_values_map, on=column)
    new_values_df = new_values_df.drop(columns=[column])
    new_values_df = new_values_df.rename(columns={tmp_col_name: column})
    return new_values_df


def preprocess_data(project_config):
    full_df = xd.read_csv(project_config['data_sources']['raw_data'], names=['session', '_t', 'items', '_c', ])

    reset_ids = project_config['reset_ids']
    if reset_ids:
        reset_sessions = reset_values(full_df, 'session')
        full_df = reset_values(reset_sessions, 'items')

    full_df = get_unique_day_values(full_df)

    preprocessed_df = full_df[['session', '_t', 'items', 'day', '_c', ]]
    preprocessed_df = preprocessed_df.sort_values(by=['session', '_t'])
    preprocessed_df.to_csv(project_config['data_sources']['prep_data'], index=False, header=False)


def get_unique_day_values(df):
    df['day'] = df['_t'].str.slice(start=0, stop=10)
    return df
