import numpy as np
from basic_application import with_exception


class DataPointError(Exception):
    pass


class DataPoint:
    @with_exception(DataPointError)
    def __init__(self, dataset, dataset_future=None):
        self.data = dataset
        self.data_f = dataset_future
        self.current_index = max(self.data.index)
        self.offset = len(dataset)

    @with_exception(DataPointError)
    def get_current_ts(self):
        return self.current_index

    @with_exception(DataPointError)
    def get_price(self, name, cursor=None):
        if cursor is None:
            cursor = self.current_index
        val = self.data.loc[cursor, name]
        return val

    @with_exception(DataPointError)
    def get_prices(self, name):
        data = self.data.loc[:, name]
        return data

    @with_exception(DataPointError)
    def get_future_prices(self, name, cursor=None):
        """Возвращает будущие точки"""
        data = self.data_f.loc[:, name]
        return data

    @with_exception(DataPointError)
    def get_timestamps(self):
        return self.data.index.values

    @with_exception(DataPointError)
    def get_current_data(self):
        return self.data

    @with_exception(DataPointError)
    def get_future_data(self):
        return self.data_f

    @with_exception(DataPointError)
    def get_last_diffs(self, num):
        data = self.get_current_data()
        col_idx = np.argmax(data.columns == 'lowest_ask')
        diffs = data.diff(axis=0).iloc[-num:, col_idx]
        return diffs.values




class DataPoint_old:
    @with_exception(DataPointError)
    def __init__(self, dataset, offset, future_offset):
        self.data = dataset
        self.offset = offset
        self.future_offset = future_offset
        self.current_index = self.offset - 1

    @with_exception(DataPointError)
    def get_current_ts(self):
        val = self.data.index[self.current_index]
        return val

    @with_exception(DataPointError)
    def get_price(self, name, cursor=None):
        if cursor is None:
            cursor = self.current_index
        col_mask = self.data.columns == name
        val = self.data.iloc[cursor, col_mask].values[0]
        return val

    @with_exception(DataPointError)
    def get_prices(self, name):
        col_mask = self.data.columns == name
        data = self.data.iloc[:self.current_index + 1, col_mask]
        return data

    @with_exception(DataPointError)
    def get_future_prices(self, name, cursor=None):
        """Возвращает будущие точки"""
        if cursor is None:
            cursor = self.current_index

        col_mask = self.data.columns == name
        bound_low = cursor + 1
        bound_hi = bound_low + self.future_offset
        data = self.data.iloc[bound_low:bound_hi, col_mask]
        return data

    @with_exception(DataPointError)
    def get_timestamps(self):
        idxs = self.data.index[:self.current_index + 1].values
        return idxs

    @with_exception(DataPointError)
    def get_current_data(self):
        data = self.data.iloc[:self.current_index + 1, :]
        return data

    @with_exception(DataPointError)
    def get_future_data(self):
        data = self.data.iloc[self.current_index + 1:, :]
        return data

    @with_exception(DataPointError)
    def get_last_diffs(self, num):
        data = self.get_current_data()
        col_idx = np.argmax(data.columns == 'lowest_ask')
        diffs = data.diff(axis=0).iloc[-num:, col_idx]
        return diffs.values