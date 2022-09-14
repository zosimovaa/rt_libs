from .abstract_feature import AbstractFeatureWithHistory
import logging
import numpy as np
import json


logger = logging.getLogger(__name__)


class OrderbookAsksFeature(AbstractFeatureWithHistory):
    LEVELS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

    def __init__(self, context, source="asks"):
        super().__init__(context)
        self.source = source

    def _build_point(self, dp, cursor=None):
        dp = self.context.data_point

        if cursor is None:
            cursor = dp.get_current_ts()

        cursor_prev = cursor - dp.period

        data_raw = dp.get_value(self.source, cursor=cursor_prev)
        orderbook_prev = json.loads(data_raw.replace("\'", "\""))

        data_raw = dp.get_value(self.source, cursor=cursor)
        orderbook = json.loads(data_raw.replace("\'", "\""))

        lowest_ask = np.min(np.array(list(map(float, orderbook.keys()))))

        asks_vol_prev = self._get_volumes(orderbook_prev, lowest_ask)
        asks_vol = self._get_volumes(orderbook, lowest_ask)

        asks_rel = np.array((asks_vol / asks_vol_prev) - 1)

        clipped = np.clip(asks_rel, -5, 5)

        return clipped

    def _get_volumes(self, asks, lowest_ask):
        if len(asks):
            keys_asks = np.array(list(map(float, asks.keys())))
            vols_asks = np.array(list(map(float, asks.values())))
            asks_ = []

            for coef in self.LEVELS:
                mask_ask = keys_asks < lowest_ask * (1 + coef)
                asks_.append(sum(vols_asks[mask_ask]))
        else:
            asks_ = np.zeros(len(self.LEVELS))

        return np.array(asks_)


class OrderbookBidsFeature(AbstractFeatureWithHistory):
    LEVELS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

    def __init__(self, context, source="bids"):
        super().__init__(context)
        self.source = source

    def _build_point(self, dp, cursor=None):
        dp = self.context.data_point

        if cursor is None:
            cursor = dp.get_current_ts()

        cursor_prev = cursor - dp.period

        data_raw = dp.get_value(self.source, cursor=cursor_prev)
        orderbook_prev = json.loads(data_raw.replace("\'", "\""))

        data_raw = dp.get_value(self.source, cursor=cursor)
        orderbook = json.loads(data_raw.replace("\'", "\""))

        highest_bid = np.max(np.array(list(map(float, orderbook.keys()))))

        bids_vol_prev = self._get_volumes(orderbook_prev, highest_bid)
        bids_vol = self._get_volumes(orderbook, highest_bid)

        bids_rel = (bids_vol / bids_vol_prev) - 1

        clipped = np.clip(bids_rel, -5, 5)

        return clipped

    def _get_volumes(self, bids, highest_bid):
        if len(bids):
            keys_bids = np.array(list(map(float, bids.keys())))
            vols_bids = np.array(list(map(float, bids.values())))

            bids_ = []

            for coef in self.LEVELS:
                mask_bid = keys_bids > highest_bid * (1 - coef)
                bids_.append(sum(vols_bids[mask_bid]))
        else:
            bids_ = np.zeros(len(self.LEVELS))

        return np.array(bids_)