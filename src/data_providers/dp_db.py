import pytz
import logging
import datetime
import numpy as np
import pandas as pd

from .dp_abstract import AbstractDataProvider
from .dp_abstract import DataProviderError
from basic_application import with_exception
from .clickhouse_connector import ClickHouseConnector

logger = logging.getLogger(__name__)

QUERY = """
WITH
    toUInt32(%(period)s) AS p_period,
    toUInt32(%(start)s) AS ts_min,
    toUInt32(%(end)s+p_period) AS ts_max,
    toUInt32(%(step)s) AS step

	SELECT 
		period_gr + ts_min as ts,
		pair_gr as pair,
		--max(ts_local_gr)+ts_min as ts,	
		
		--anyLast(highest_bid_gr) as open_price,
		--any(highest_bid_gr) as close_price, 
		
		--min(highest_bid_gr) as min_price,
		--max(highest_bid_gr) as max_price,
		--anyLast(highest_bid_gr) as avg_price,
		
		anyLast(highest_bid_gr) as highest_bid,
		anyLast(lowest_ask_gr) as lowest_ask


		
	FROM (
        SELECT 
            pair as pair_gr, 
            ts-ts_min as ts_local_gr, 
            CEIL((ts-ts_min) / p_period) * p_period AS period_gr, 
            lowest_ask as lowest_ask_gr, 
            highest_bid as highest_bid_gr
        FROM orderbook
        WHERE ts BETWEEN ts_min+1 AND ts_max
        ORDER BY ts DESC
        ) AS TB1
        
        RIGHT JOIN (SELECT arrayJoin (range(p_period, toUInt32(ts_max - ts_min + p_period), p_period)) AS ts_main) AS tb2 ON TB1.period_gr = tb2.ts_main
        
    GROUP BY pair_gr, period_gr
	ORDER BY pair_gr, period_gr
        
"""

class DbDataProvider(AbstractDataProvider):

    def __init__(self, conn):
        self.conn = conn
        self.query = QUERY

    @with_exception(DataProviderError)
    def get(self, start, end, period, pairs, step=60):
        params = self._build_params(start, end, period, step)

        self.conn.cursor.execute(self.query, parameters=params)
        raw_data = self.conn.cursor.fetchall()
        col_names = [col[0] for col in self.conn.cursor.columns_with_types]

        data = self._transform(raw_data, col_names)
        if len(pairs):
            data = data.loc[data.loc[:, "pair"].isin(pairs), :]

        return data

    def _build_params(self, start, end, period, step):
        params = {
            "period": period,
            "start": self.date_to_unix_ts_in_utc(start),
            "end": self.date_to_unix_ts_in_utc(end),
            "step": step,
        }
        return params

    @staticmethod
    def _transform(raw_data, col_names):
        data = pd.DataFrame(raw_data, columns=col_names)
        data["ts"] = data["ts"].astype(int)
        data["lowest_ask"] = data["lowest_ask"].astype(np.float32)
        data["highest_bid"] = data["highest_bid"].astype(np.float32)
        data = data.set_index("ts")



        data = data.replace(to_replace=0, method='ffill')

        return data