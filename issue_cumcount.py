import cudf
import pandas as pd

from rmm._cuda.gpu import runtimeGetVersion, driverGetVersion

# RecSys challenge 2015 dataset: https://www.kaggle.com/chadgostopp/recsys-challenge-2015

print("cudf version: ", cudf.__version__)
print("cuda runtime version: ", runtimeGetVersion())
print("drivers version: ", driverGetVersion())


DF_20_ROWS = 'archive/yoochoose-data/20_rows.dat'
DF_424_ROWS = 'archive/yoochoose-data/424_rows.dat'

pandas_20 = pd.read_csv(DF_20_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])
pandas_424 = pd.read_csv(DF_424_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])

rapids_20 = cudf.read_csv(DF_20_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])
rapids_424 = cudf.read_csv(DF_424_ROWS, names=['session_id', 'timestamp', 'item_id', 'category'])


for df in [pandas_20, rapids_20, pandas_424, rapids_424]:
    df['item_pos'] = df.groupby('session_id').cumcount()
    print(df.loc[df['session_id'] == 121])
