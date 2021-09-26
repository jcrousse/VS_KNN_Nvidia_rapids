import cudf
import pandas as pd

DF_20_ROWS = 'archive/yoochoose-data/20_rows.dat'
DF_424_ROWS = 'archive/yoochoose-data/424_rows_sorted.dat'

pandas_20 = pd.read_csv(DF_20_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])
pandas_424 = pd.read_csv(DF_424_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])

rapids_20 = cudf.read_csv(DF_20_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])
rapids_424 = cudf.read_csv(DF_424_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])

# pandas_424_sorted = pandas_424.sort_values(by=['session_id', 'timestamp'], ascending=[True, True])
# pandas_424_sorted.to_csv('archive/yoochoose-data/424_rows_sorted.dat', index=False, header=False)
# test = pandas_424_sorted['timestamp'] == pandas_424['timestamp']




for df in [pandas_20, rapids_20, pandas_424, rapids_424]:
    df['cumcount'] = df.groupby('session_id').cumcount()
    print(df.loc[df['session_id'] == 121])
