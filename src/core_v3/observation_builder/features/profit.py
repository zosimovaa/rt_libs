"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class ProfitFeature(BaseFeature):
    """Profit calculation"""
    def __init__(self, *args, **kwargs):
        self.price = kwargs.pop("price", "highest_bid")
        super().__init__(*args, **kwargs)

    def _get(self):
        is_open = self.context.get("is_open")
        data_point = self.context.get("data_point")
        trade = self.context.get("trade")

        if is_open:
            timestamps = data_point.get_indexes(period=self.period)
            mask = (timestamps >= trade.open_ts) & (timestamps <= trade.close_ts)

            values = data_point.get_values(self.price, period=self.period)

            feature = values / trade.open_price - 1 - trade.market_fee
            feature = feature * mask
        else:

            feature = np.zeros(data_point.observation_len)

        return feature


class ProfitDiffFeature(BaseFeature):
    """Profit calculation"""

    def __init__(self, *args, **kwargs):
        self.price = kwargs.pop("price", "highest_bid")
        super().__init__(*args, **kwargs)
        raise Exception('Требует проверки и доработки')

    def _get(self):
        trade_state = self.context.get("is_open")
        data_point = self.context.get("data_point")
        trade = self.context.get("trade")

        if trade_state:
            timestamps = data_point.get_indexes(period=self.period)
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)

            # в DiffFeature нужна +1 точка, чтобы рассчитать разницу.
            values = data_point.get_values(self.price, period=self.period, num=data_point.observation_len + 1)

            feature = values / trade.open_price - 1 - trade.market_fee
            feature = np.diff(feature) * mask

        else:
            feature = np.zeros(data_point.OBSERVATION_LEN)

        return feature
