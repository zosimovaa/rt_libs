"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class ProfitFeature2D(BaseFeature):
    """Profit calculation"""
    def __init__(self, context, step_factor=1, scale_output=30):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        is_open = self.context.get("is_open")
        data_point = self.context.data_point
        trade = self.context.get("trade")

        if is_open:
            timestamps = data_point.get_points(step_factor=self.step_factor)
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
            current_rates = data_point.get_values("highest_bid", step_factor=self.step_factor)
            current_rates = current_rates[self.step_factor - 1::self.step_factor]
            profit = current_rates / trade.open_price - 1 - trade.market_fee
            profit = profit * mask * self.scale_output
        else:
            profit = np.zeros(data_point.OBSERVATION_LEN)

        return profit.reshape(-1, 1)


class ProfitDiffFeature2D(BaseFeature):
    """Profit calculation"""

    def __init__(self, context, step_factor=1, scale_output=100):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        trade_state = self.context.get("is_open")
        data_point = self.context.data_point
        trade = self.context.get("trade")

        if trade_state:
            timestamps = data_point.get_points(step_factor=self.step_factor, num=data_point.OBSERVATION_LEN + 1)
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
            current_rates = data_point.get_values("highest_bid", step_factor=self.step_factor, num=data_point.OBSERVATION_LEN + 1)
            current_rates = current_rates[self.step_factor - 1::self.step_factor]
            profit = current_rates / trade.open_price - 1 - trade.market_fee
            profit = profit * mask

            profit_diff = np.diff(profit) * self.scale_output

        else:
            profit_diff = np.zeros(data_point.OBSERVATION_LEN)

        return profit_diff.reshape(-1, 1)
