"""
Вариант дата провайдера с выгрузкой информации из таблиц orderbook и trades

"""
import logging

import numpy as np
import pandas as pd
from basic_application import with_exception

from .dp_abstract import AbstractDataProvider
from .dp_abstract import DataProviderError, TooManyGapsError, UpToDateError

logger = logging.getLogger(__name__)

QUERY = """
WITH

    toUInt32(%(period)s) AS p_period,
    toUInt32(%(start)s) AS ts_min_,
    toUInt32(%(end)s) AS ts_max,
    %(pair)s AS p_pair,

    toUInt32(ts_min_ - p_period) AS ts_min

SELECT 
	tb_ts.ts as ts,
	FROM_UNIXTIME(toUInt32(tb_ts.ts)) as dt,
	lowest_ask,
	highest_bid,
	asks,
	bids,
	buy_vol,
	sell_vol,
	toFloat32(buy_vol+sell_vol) as total_col,
	buy_num,
	sell_num,
	toUInt32(buy_num+sell_num) as total_num

FROM 
	(SELECT 
		tb_orderbook.ts as ts,
		tb_orderbook.lowest_ask as lowest_ask,
		tb_orderbook.highest_bid as highest_bid,
		tb_orderbook.asks as asks,
		tb_orderbook.bids as bids,

		tb_trades.buy_vol as buy_vol,
		tb_trades.sell_vol as sell_vol,
		tb_trades.buy_num as buy_num,
		tb_trades.sell_num as sell_num

	FROM     


		(SELECT 
			period_gr + ts_min as ts,
			toFloat32(any(lowest_ask)) as lowest_ask,
			toFloat32(any(highest_bid)) as highest_bid,
			any(asks) as asks,
			any(bids) as bids

		FROM

			(SELECT
				ts - ts_max AS ts_group,
				CEIL((`ts`-`ts_min`) / p_period) * p_period  AS `period_gr`, 
				lowest_ask,
				highest_bid, 
				asks,
				bids
			FROM orderbook 
			WHERE ts > ts_min and ts <= ts_max and pair=p_pair
			ORDER BY ts_group DESC
			) AS tb_orderbook_ungrouped
		GROUP BY period_gr
		) as tb_orderbook

	FULL JOIN

	(SELECT
		toUInt32(period + ts_min) AS ts,
		toFloat32(SUM(buy_vol)) as buy_vol,
		toFloat32(SUM(sell_vol)) as sell_vol,

		SUM(buy_num) as buy_num,
		SUM(sell_num) as sell_num

	FROM
		(SELECT
		    CEIL((`ts`-`ts_min`) / p_period) * p_period  AS `period`,
		    --`type` as `type_gr`,
		    --`rate` as rate_gr,
		    --`amount` as `amount_gr`,
		    --`total` as `total_ts`,

		    if(`type`='buy', total, 0) as `buy_vol`,
		  	if(`type`='buy', 1, 0) as `buy_num`,
		    if(`type`='sell', total, 0) as `sell_vol`,
		    if(`type`='sell', 1, 0) as `sell_num`

		FROM trades
		WHERE ts > ts_min AND ts <= ts_max AND pair=p_pair) AS tb_trades_ungrouped
	GROUP BY period
	) AS tb_trades ON tb_trades.ts = tb_orderbook.ts

	ORDER BY ts ASC ) AS tb_ob_trades

	RIGHT JOIN

	(SELECT arrayJoin (range(toUInt32(ts_min+p_period), toUInt32(ts_max+p_period), p_period)) AS ts) as tb_ts on tb_ts.ts = tb_ob_trades.ts
ORDER BY tb_ts.ts ASC"""


class DbDataProviderV2(AbstractDataProvider):
    GAPS_THRESHOLD = 0.3
    HISTORY_DEPTH = 100


    DICT_COLUMNS = ["asks", "bids"]
    STR_COLUMNS = ["pair"]
    INT_COLUMNS = ["ts", "buy_num", "sell_num"]
    WO_CONVERT = ["dt"]
    INDEX_COL = "ts"

    QUERY = QUERY

    def __init__(self, conn):
        self.conn = conn
        self.raw_data = None
        self.col_names = None



    @with_exception(DataProviderError)
    def get(self, start, end, period, pair=None, raise_errors=False):
        params = self._build_params(start, end, period, pair)
        self.conn.cursor.execute(self.QUERY, parameters=params)
        self.raw_data = self.conn.cursor.fetchall()
        self.col_names = [col[0] for col in self.conn.cursor.columns_with_types]

        data = self._transform(self.raw_data, self.col_names)
        if raise_errors:
            self._check_data(data)

        data = self._fill_gaps(data)

        return data

    def get_by_periods(self, ts, period, num_of_period, pair=None, raise_errors=False) -> pd.DataFrame:
        ts_end = self.date_to_unix_ts_in_utc(ts)
        start = ts_end - period * (num_of_period - 1)
        end = ts_end
        result = self.get(start, end, period, pair=pair, raise_errors=raise_errors)
        return result

    def _build_params(self, start, end, period, pair):
        params = dict()
        params["start"] = self.date_to_unix_ts_in_utc(start)
        params["end"] = self.date_to_unix_ts_in_utc(end)
        params["period"] = period
        params["pair"] = pair
        return params

    def _transform(self, raw_data, col_names):
        data = pd.DataFrame(raw_data, columns=col_names)

        for col in data.columns:
            if col in self.STR_COLUMNS:
                data[col] = data[col].astype(str)
            elif col in self.INT_COLUMNS:
                data[col] = data[col].astype(int)
            elif col in self.DICT_COLUMNS:
                pass
            else:
                if col not in self.WO_CONVERT:
                    data[col] = data[col].astype(np.float32)

        if self.INDEX_COL in data.columns:
            data = data.set_index(self.INDEX_COL)

        return data

    def _check_data(self, data):
        zero_counts = sum(data["highest_bid"] == 0)
        data_length = len(data["highest_bid"])

        if zero_counts / data_length > self.GAPS_THRESHOLD:
            raise TooManyGapsError(zero_counts, data_length)

        highest_bid = data.loc[max(data.index), "highest_bid"]
        if not highest_bid:
            raise UpToDateError()

    def _fill_gaps(self, data):
        data['lowest_ask'] = data['lowest_ask'].replace(to_replace=0, method='bfill')
        data['highest_bid'] = data['highest_bid'].replace(to_replace=0, method='bfill')
        data['asks'] = data['asks'].replace(to_replace=dict(), method='bfill')
        data['bids'] = data['bids'].replace(to_replace=dict(), method='bfill')

        return data
