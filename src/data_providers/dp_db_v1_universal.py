"""
Универсальный дата провайдер, но применения так и не нашел из-за своей универсальности
"""
import logging

import numpy as np
import pandas as pd
from basic_application import with_exception

from .dp_abstract import AbstractDataProvider
from .dp_abstract import DataProviderError

logger = logging.getLogger(__name__)


class DbDataProviderUniversal(AbstractDataProvider):
    DICT_COLUMNS = ["asks", "bids"]
    STR_COLUMNS = ["pair"]
    INT_COLUMNS = ["ts", "buy_num", "sell_num"]
    INDEX_COL = "ts"

    def __init__(self, conn, query):
        self.conn = conn
        self.query = query
        self.raw_data = None
        self.col_names = None

    @with_exception(DataProviderError)
    def get(self, params):
        params = self._build_params(params.copy())
        self.conn.cursor.execute(self.query, parameters=params)
        self.raw_data = self.conn.cursor.fetchall()
        self.col_names = [col[0] for col in self.conn.cursor.columns_with_types]

        data = self._transform(self.raw_data, self.col_names)

        return data

    def _build_params(self, params):
        if "start" in params:
            params["start"] = self.date_to_unix_ts_in_utc(params["start"])

        if "end" in params:
            params["end"] = self.date_to_unix_ts_in_utc(params["end"])

        return params

    def _transform(self, raw_data, col_names):
        data = pd.DataFrame(raw_data, columns=col_names)
        data = data.bfill()

        for col in data.columns:
            if col in self.STR_COLUMNS:
                data[col] = data[col].astype(str)
            elif col in self.INT_COLUMNS:
                data[col] = data[col].astype(int)
            elif col in self.DICT_COLUMNS:
                pass
            else:
                data[col] = data[col].astype(np.float32)

        if self.INDEX_COL in data.columns:
            data = data.set_index(self.INDEX_COL)

        return data
