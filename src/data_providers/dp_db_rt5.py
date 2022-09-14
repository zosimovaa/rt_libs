"""
Вариант дата провайдера с выгрузкой информации из таблиц orderbook и trades

"""
import logging

import numpy as np
import pandas as pd
from basic_application import with_exception

from data_providers.abstract_provider import AbstractDataProvider
from data_providers.errors import DataProviderError, TooManyGapsError, UpToDateError

logger = logging.getLogger(__name__)

QUERY = """
WITH
    toUInt32(%(ts)s) AS ts_max,
    toUInt32(%(depth)s) AS depth,
    toUInt32(%(period)s) AS p_period,
    %(pair)s AS p_pair,
   
    toUInt32(ts_max - depth * p_period) AS ts_min


SELECT
	# Соединение данных и индекса
	tb_ts.ts as ts,
	FROM_UNIXTIME(toUInt32(tb_ts.ts)) as dt,
	lowest_ask,
	highest_bid,
	asks,
	bids,
	buy_vol,
	sell_vol,
	toFloat32(buy_vol+sell_vol) as total_vol,
	buy_num,
	sell_num,
	toUInt32(buy_num+sell_num) as total_num


FROM 
	# Данные
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
			WHERE ts > ts_min and ts <= ts_max and symbol=p_pair
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
		    if(`takerSide`='BUY', quantity, 0) as `buy_vol`,
		  	if(`takerSide`='BUY', 1, 0) as `buy_num`,
		    if(`takerSide`='SELL', quantity, 0) as `sell_vol`,
		    if(`takerSide`='SELL', 1, 0) as `sell_num`

		FROM trades
		WHERE ts > ts_min AND ts <= ts_max AND symbol=p_pair) AS tb_trades_ungrouped
	GROUP BY period
	) AS tb_trades ON tb_trades.ts = tb_orderbook.ts


	ORDER BY ts ASC ) AS tb_ob_trades

	RIGHT JOIN
	# Индекс
	(SELECT arrayJoin (range(toUInt32(ts_min+p_period), toUInt32(ts_max+p_period), p_period)) AS ts) as tb_ts on tb_ts.ts = tb_ob_trades.ts
ORDER BY tb_ts.ts ASC
"""


class DbDataProviderRT5(AbstractDataProvider):
    #GAPS_THRESHOLD = 0.3

    DICT_COLUMNS = ["asks", "bids"]
    STR_COLUMNS = []
    INT_COLUMNS = ["ts", "buy_num", "sell_num", "total_num"]
    WO_CONVERT = ["dt"]
    INDEX_COL = "ts"

    QUERY = QUERY

    def __init__(self, conn, gaps_threshold=0.):
        self.conn = conn
        self.raw_data = None
        self.col_names = None
        self.gaps_threshold = gaps_threshold

    @with_exception(DataProviderError)
    def get(self, ts, period, num_of_periods, pair, raise_errors=False, fill_gaps=True):
        params = self._build_params(ts, period, num_of_periods, pair)

        self.conn.cursor.execute(self.QUERY, parameters=params)
        self.raw_data = self.conn.cursor.fetchall()
        self.col_names = [col[0] for col in self.conn.cursor.columns_with_types]

        data = self._transform(self.raw_data, self.col_names)
        if raise_errors:
            self._check_data(data)

        if fill_gaps:
            data = self._fill_gaps(data)

        return data

    def _build_params(self, ts, period, num_of_periods, pair):
        params = dict()
        params["ts"] = self.date_to_unix_ts_in_utc(ts)
        params["period"] = period
        params["depth"] = num_of_periods
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
        highest_bid = data.loc[max(data.index), "highest_bid"]
        if not highest_bid:
            raise UpToDateError()

        zero_counts = sum(data["highest_bid"] == 0)
        data_length = len(data["highest_bid"])

        if zero_counts / data_length > self.gaps_threshold:
            raise TooManyGapsError(zero_counts, data_length)

    def _fill_gaps(self, data):
        data['lowest_ask'] = data['lowest_ask'].replace(to_replace=0, method='bfill')
        data['highest_bid'] = data['highest_bid'].replace(to_replace=0, method='bfill')
        data['asks'] = data['asks'].replace(to_replace=dict(), method='bfill')
        data['bids'] = data['bids'].replace(to_replace=dict(), method='bfill')
        return data


if __name__ == "__main__":
    print("go!")