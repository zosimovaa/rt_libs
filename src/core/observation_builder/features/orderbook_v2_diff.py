from .abstract_feature import AbstractFeatureWithHistory
import logging
import numpy as np
import pandas as pd
import json


logger = logging.getLogger(__name__)


class OrderbookDiffFeature(AbstractFeatureWithHistory):
    LEVELS = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02]

    def __init__(self, context, level=0.0025):
        super().__init__(context)
        self.level = level

    def _build_point(self, dp, cursor=None):
        dp = self.context.data_point

        if cursor is None:
            cursor = dp.get_current_ts()

        n1 = dp.hist_len

        # 1. Получить данные по orderbook
        data = dp.get_values2(cursor=cursor, num=n1)

        # 2. Построить объемы по asks и bids
        asks_vols, bids_vols = self._convert_orderbook(data)

        # 3. Рассчитать разницу
        asks_diff = np.diff(asks_vols)
        bids_diff = np.diff(bids_vols)

        # 4. Вычисление разницы между asks и bids (diff)
        diff = asks_diff - bids_diff

        # 5. Значение точки данных
        feature = diff.mean()
        return feature

    def _get_volume(self, asks, bids, lowest_ask, highest_bid):
        keys_asks = np.array(list(map(float, asks.keys())))
        vols_asks = np.array(list(map(float, asks.values())))

        keys_bids = np.array(list(map(float, bids.keys())))
        vols_bids = np.array(list(map(float, bids.values())))

        mask_ask = keys_asks < lowest_ask * (1 + self.level)
        ask_vol = sum(vols_asks[mask_ask])

        mask_bid = keys_bids > highest_bid * (1 - self.level)
        bid_vol = sum(vols_bids[mask_bid])

        return ask_vol, bid_vol

    def _convert_orderbook(self, data):
        asks_vol_arr, bids_vol_arr = [], []
        idxs = data.index.values

        index_length = len(idxs)
        for i in range(index_length - 1):
            idx_curr = idxs[i + 1]
            asks_curr = json.loads(data.loc[idx_curr, "asks"].replace("\'", "\""))
            bids_curr = json.loads(data.loc[idx_curr, "bids"].replace("\'", "\""))

            idx_prev = idxs[i]
            asks_prev = json.loads(data.loc[idx_prev, "asks"].replace("\'", "\""))
            bids_prev = json.loads(data.loc[idx_prev, "bids"].replace("\'", "\""))

            if len(asks_curr) and len(asks_prev) and len(bids_curr) and len(bids_prev):
                lowest_ask = np.min(np.array(list(map(float, asks_prev.keys()))))
                highest_bid = np.max(np.array(list(map(float, bids_prev.keys()))))

                ask_curr_vol, bid_curr_vol = self._get_volume(asks_curr, bids_curr, lowest_ask, highest_bid)
                asks_vol_arr.append(ask_curr_vol)
                bids_vol_arr.append(bid_curr_vol)
            else:

                asks_vol_arr.append(0)
                bids_vol_arr.append(0)

        return np.array(asks_vol_arr), np.array(bids_vol_arr)
