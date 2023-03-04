import json
import logging
import numpy as np

from .base_precompute import BasePrecompute

logger = logging.getLogger(__name__)


class PrecomputeOrderbookDiffFeature(BasePrecompute):

    def process(self, data, level):
        # 1. Построить объемы по asks и bids
        asks_vols, bids_vols = self._convert_orderbook(data, level)

        # 5. Значение точки данных
        asks_feature = np.concatenate([np.zeros(1), asks_vols])
        bids_feature = np.concatenate([np.zeros(1), bids_vols])

        data["asks_" + str(level)] = asks_feature
        data["bids_" + str(level)] = bids_feature

        return data

    def _get_volume(self, asks, bids, lowest_ask, highest_bid, level):
        keys_asks = np.array(list(map(float, asks.keys())))
        vols_asks = np.array(list(map(float, asks.values())))

        keys_bids = np.array(list(map(float, bids.keys())))
        vols_bids = np.array(list(map(float, bids.values())))

        mask_ask = keys_asks < lowest_ask * (1 + level)
        ask_vol = sum(vols_asks[mask_ask])

        mask_bid = keys_bids > highest_bid * (1 - level)
        bid_vol = sum(vols_bids[mask_bid])

        return ask_vol, bid_vol

    def _convert_orderbook(self, data, level):
        asks_vol_arr, bids_vol_arr = [], []
        idxs = data.index.values

        index_length = len(idxs)
        for i in range(index_length - 1):
            idx_curr = idxs[i + 1]
            idx_prev = idxs[i]
            try:
                # Быстрый вариант для загрузки из файла
                asks_curr = json.loads(data.loc[idx_curr, "asks"].replace("\'", "\""))  #
                bids_curr = json.loads(data.loc[idx_curr, "bids"].replace("\'", "\""))  #

                asks_prev = json.loads(data.loc[idx_prev, "asks"].replace("\'", "\""))  #
                bids_prev = json.loads(data.loc[idx_prev, "bids"].replace("\'", "\""))  #

            except Exception as e:
                # Быстрый вариант для загрузки из БД
                asks_curr = data.loc[idx_curr, "asks"]  # json.loads(data.loc[idx_curr, "asks"].replace("\'", "\"")) #
                bids_curr = data.loc[idx_curr, "bids"]  # json.loads(data.loc[idx_curr, "bids"].replace("\'", "\"")) #

                asks_prev = data.loc[idx_prev, "asks"]  # json.loads(data.loc[idx_prev, "asks"].replace("\'", "\"")) #
                bids_prev = data.loc[idx_prev, "bids"]  # json.loads(data.loc[idx_prev, "bids"].replace("\'", "\"")) #

            if len(asks_curr) and len(asks_prev) and len(bids_curr) and len(bids_prev):
                lowest_ask = np.min(np.array(list(map(float, asks_prev.keys()))))
                highest_bid = np.max(np.array(list(map(float, bids_prev.keys()))))

                ask_curr_vol, bid_curr_vol = self._get_volume(asks_curr, bids_curr, lowest_ask, highest_bid, level)
                asks_vol_arr.append(ask_curr_vol)
                bids_vol_arr.append(bid_curr_vol)
            else:

                asks_vol_arr.append(0)
                bids_vol_arr.append(0)

        return np.array(asks_vol_arr), np.array(bids_vol_arr)