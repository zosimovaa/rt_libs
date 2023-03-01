"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .abstract_feature import AbstractFeature

logger = logging.getLogger(__name__)


class ProfitFeature2D(AbstractFeature):
    """Profit calculation"""
    def __init__(self, context, step_factor=(1,), scale_output=30):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        trade_state = self.context.get("is_open", domain="Trade")
        data_point = self.context.data_point
        trade = self.context.trade
        obs = []
        for sf in self.step_factor:
            if trade_state:
                timestamps = data_point.get_points(step_factor=sf)
                mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
                current_rates = data_point.get_values("highest_bid", step_factor=sf)
                current_rates = current_rates[sf - 1::sf]
                profit = current_rates / trade.open_price - 1 - trade.market_fee
                profit = profit * mask * self.scale_output
            else:
                profit = np.zeros(data_point.observation_len)

            obs.append(profit.reshape(-1, 1))

        feature = np.concatenate([*obs], axis=1)
        return feature


class ProfitDiffFeature2D(AbstractFeature):
    """Profit calculation"""

    def __init__(self, context, step_factor=(1,), scale_output=100):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        trade_state = self.context.get("is_open", domain="Trade")
        data_point = self.context.data_point
        trade = self.context.trade
        obs = []
        for sf in self.step_factor:
            if trade_state:
                timestamps = data_point.get_points(step_factor=sf, num=data_point.observation_len + 1)
                mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
                current_rates = data_point.get_values("highest_bid", step_factor=sf, num=data_point.observation_len + 1)
                current_rates = current_rates[sf - 1::sf]
                profit = current_rates / trade.open_price - 1 - trade.market_fee
                profit = profit * mask

                profit_diff = np.diff(profit) * self.scale_output

            else:
                profit_diff = np.zeros(data_point.observation_len)

            obs.append(profit_diff.reshape(-1, 1))

        feature = np.concatenate([*obs], axis=1)
        return feature