import cudf
import cupy as cp
from cupyx.time import repeat
import random

data_path = "../data/train_set.dat"
col_names = ['session_id', 'timestamp', 'item_id', 'category']
cp_data = cudf.read_csv(data_path,  names=col_names)[['session_id', 'timestamp',  'item_id']]\
    .sort_values(by='timestamp', ascending=False)\
    .drop_duplicates(subset=['session_id', 'item_id'], keep='first')

item_data = cp_data.reset_index().rename(columns={'index': 'item_index'})

sessions = cp_data['session_id'].unique()

session_cp = cp_data['session_id'].values
item_cp = cp_data['item_id'].values


def get_sub_index(items_scope, n_elements, max_spi):
    print(n_elements)
    data_slice = item_data[item_data['item_index'].isin(items_scope)]
    data_slice = data_slice.sort_values(by=['item_index', 'timestamp'], ascending=[True, False]).reset_index()
    if n_elements > 1:
        data_slice['n_col'] = data_slice.groupby(['item_index']).cumcount()
        if n_elements == max_spi:
            data_slice = data_slice[data_slice['n_col'] <= max_spi]
        index_array = data_slice.pivot(index='item_index', columns='n_col', values=['session_id'])
    else:
        index_array = data_slice.set_index('item_index')[['session_id']]
    return index_array.values


def create_indices(max_spi=100):
    n_session_per_item_df = cp_data\
        .groupby('item_id')\
        .agg({'session_id': 'count'})\
        .sort_index()
    item_to_idx = n_session_per_item_df.reset_index().reset_index().set_index('item_id').to_pandas()['index'].to_dict()

    n_session_per_item = n_session_per_item_df.values.flatten()
    n_session_per_item[n_session_per_item > max_spi] = max_spi
    count_values = cp.unique(n_session_per_item).tolist()
    index_table = {}
    position_dfs = []
    for cnt_value in count_values[0:10]:
        items_scope = cp.where(n_session_per_item == cnt_value)[0] + 1
        index_table[cnt_value] = get_sub_index(items_scope, cnt_value, max_spi)
        position_in_index = cudf.DataFrame(
            data={'position_in_index': list(range(len(items_scope)))},
            index=items_scope)
        position_dfs.append(position_in_index)
    position_df = cudf.concat(position_dfs).sort_index()



if __name__ == '__main__':
    create_indices()
