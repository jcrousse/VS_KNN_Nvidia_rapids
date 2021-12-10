import pandas as pd
import cudf
import cupy as cp

from vs_knn.col_names import SESSION_ID, TIMESTAMP, ITEM_ID, DAY, position_to_name


def read_dataset(dataset, project_config, reader='pandas'):

    column_names = get_column_names(project_config, dataset)

    if dataset == 'raw_data':
        data_path = project_config['data_sources']['raw_data']
        usecols = list(project_config['data_schema'].values())
    elif dataset == 'prep_data':
        data_path = project_config['data_sources']['prep_data']
        usecols = list(range(len(column_names)))
    elif dataset == 'train_data':
        data_path = project_config['data_sources']['train_data']
        usecols = list(range(3))
    elif dataset == 'test_data':
        data_path = project_config['data_sources']['test_data']
        usecols = list(range(3))
    else:
        raise ValueError(f" Unknown dataset name {dataset}")

    pandas_dtypes = None
    cudf_dtypes = {
        SESSION_ID: cp.dtype('int32'),
        TIMESTAMP: cp.dtype('O'),
        ITEM_ID: cp.dtype('int32')
    }
    xd = pd if reader == 'pandas' else cudf
    dtypes = pandas_dtypes if reader == 'pandas' else cudf_dtypes

    return xd.read_csv(data_path,
                       names=column_names,
                       dtype=dtypes,
                       usecols=usecols)


def get_column_names(project_config, dataset):
    column_positions = project_config['data_schema']
    columns_pos = sorted([(k, v) for k, v in column_positions.items()], key=lambda x: x[1])
    column_names = [position_to_name[col[0]] for col in columns_pos]
    if dataset == 'prep_data':
        column_names = column_names + [DAY]
    return column_names


def write_dataframe(df, dataset, project_config):
    columns = get_column_names(project_config, dataset)
    df = df[columns]
    df.to_csv(project_config['data_sources'][dataset], index=False, header=False)
