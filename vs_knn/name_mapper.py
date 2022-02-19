"""
To reduce device memory footprint whithout putting constraints on the input data format for the vsknn algorithm,
this tool transforms the input data item_id and session_id into contiguous int32 values.
Then it provides dictionaries and arrays to convert the names to ids and vice versa.
"""
import gc

import cudf
import cupy as cp
import pandas as pd

from vs_knn.col_names import SESSION_ID, ITEM_ID


class NameIdxMap:
    def __init__(self, columns_to_convert=None, skips_missings=False):
        """
        :param columns_to_convert: Columns names in input df for which an index must be created
        :param skips_missings: if True, a key is not found in name_to_idx function will be skipped instead of
        causing a KeyError.
        # todo: option to delete some of the column mappings. Since session mappings won't be used since model gets
            items in and out
        """
        self.skip_missings = skips_missings

        self.columns_to_convert = columns_to_convert if columns_to_convert else [SESSION_ID, ITEM_ID]
        self._name_to_idx_map = {col: {} for col in self.columns_to_convert}
        self._idx_to_name_map = {col: cp.array([]) for col in self.columns_to_convert}

        self._transformed_df = cudf.DataFrame()
        self._fit = False

    def build(self, df: cudf.DataFrame):

        self._validate_df(df)

        self._transformed_df = df

        for col in self.columns_to_convert:
            self._create_col_mappings(df, col)

        _padding = cudf.DataFrame(
            columns=self._transformed_df.columns,
            data={ITEM_ID: [0], SESSION_ID: [0]},
            dtype=self._transformed_df.dtypes.to_dict())

        self._transformed_df = cudf.concat([_padding, self._transformed_df])

        return self

    def name_to_idx(self, query, column_name):
        """ for every name in query array, corresponding to a value in column, returns the corresponding idx"""
        if self.skip_missings:
            return [self._name_to_idx_map[column_name][q] for q in query if q in self._name_to_idx_map[column_name]]
        else:
            return [self._name_to_idx_map[column_name][q] for q in query]

    def idx_to_name(self, query, column_name):
        """ for every idx in query array and a column name, returns the original name"""
        return self._idx_to_name_map[column_name][query]

    def _validate_df(self, df):
        missing_columns = set(self.columns_to_convert) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing expected columns {missing_columns}")

    def _create_col_mappings(self, df, col):
        """ creates name_to_idx and idx_to_name mappings for the given column 'col' """
        ordered_names = df[col].unique()

        name_to_idx_series = ordered_names.reset_index()
        name_to_idx_series['index'] = name_to_idx_series['index'] + 1
        name_to_idx_series = name_to_idx_series.set_index(col)['index']

        self._name_to_idx_map[col] = name_to_idx_series.to_pandas().to_dict()
        idx_to_name = ordered_names.to_pandas()

        idx_to_name_pad = pd.concat([
            pd.Series([idx_to_name[0]]),
            idx_to_name
        ])

        self._idx_to_name_map[col] = cp.array(idx_to_name_pad.values)

        self._transformed_df = self._transformed_df.set_index(col)\
            .join(name_to_idx_series)\
            .reset_index()\
            .drop(col, axis=1)\
            .rename(columns={'index': col})

    def get_transformed_df(self):
        return self._transformed_df

    def remove_col(self, column_name):
        del self._name_to_idx_map[column_name]
        del self._idx_to_name_map[column_name]
        gc.collect()

