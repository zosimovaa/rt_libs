"""
Module with basic features - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .abstract_class import AbstractFeature

logger = logging.getLogger(__name__)


class TradeStateFeature(AbstractFeature):
    """Retrieves the current state of a trade operation from the context"""
    def __init__(self, context):
        super().__init__(context)

    def _get(self):
        trade_state = self.context.get("is_open", domain="Trade")
        logger.debug("Get trade state() -> {}".format(trade_state))
        return trade_state


class Rates1DFeature(AbstractFeature):
    """Shows the exchange rate normalized to the current value """
    def __init__(self, context, scale_factor=10):
        super().__init__(context)
        self.scale_factor = scale_factor

    def _get(self):
        data_point = self.context.data_point
        current_price = data_point.get_value("highest_bid")
        rates = (data_point.get_values(name="highest_bid") / current_price - 1) * self.scale_factor
        logger.debug("Get rates -> current price {0} | {1}".format(current_price, rates))
        return rates


class Rates2DFactorFeature(AbstractFeature):
    """Returns a matrix with normalized rates for different periods"""
    def __init__(self, context, step_factor=(1,), scale_factor=10):
        super().__init__(context)
        self.step_factor = step_factor
        self.scale_factor = scale_factor

    def _get(self):
        data_point = self.context.data_point
        current_price = data_point.get_value("highest_bid")
        ts = data_point.get_current_index()
        obs = []
        for sf in self.step_factor:
            period = data_point.period*sf
            rates = data_point.get_values(name="highest_bid", num=data_point.obs_len*sf, as_ndarray=False)
            rates = pd.DataFrame(rates)
            rates["highest_bid"] = (rates["highest_bid"] / current_price - 1) * self.scale_factor
            rates["group"] = rates.index - ts
            rates["group"] = np.ceil(rates["group"]/period)*period
            rates = rates.groupby(["group"]).mean()
            new_idx = np.array(list(map(int, rates.index + int(ts))))
            rates = rates.set_index(new_idx)
            obs.append(rates.values.reshape(-1, 1))
        feature = np.concatenate([*obs], axis=1)
        return feature


class ProfitFeature(AbstractFeature):
    """Profit calculation"""
    def __init__(self, context, scale_factor=10):
        super().__init__(context)
        self.scale_factor = scale_factor

    def _get(self):
        timestamps = self.context.data_point.get_indexes()
        data_point = self.context.data_point
        trade = self.context.trade
        market_fee = self.context.market_fee

        trade_state = self.context.get("is_open", domain="Trade")

        if trade_state:
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
            current_rates = data_point.get_values(name="highest_bid")
            profit = current_rates / trade.open_price - 1 - market_fee
            profit = profit * mask * self.scale_factor
        else:
            profit = np.zeros(len(timestamps))
        logger.debug("Get profit -> {0}".format(profit))
        return profit

class OppositeProfitFeature(AbstractFeature):
    """Opposit Profit calculation"""
    def __init__(self, context, scale_factor=10):
        super().__init__(context)
        self.scale_factor = scale_factor

    def _get(self):
        timestamps = self.context.data_point.get_indexes()
        data_point = self.context.data_point
        opposite_trade = self.context.get("trade", domain="OppositeTrade")

        if opposite_trade.is_open:
            mask = (timestamps > opposite_trade.open_ts) & (timestamps <= opposite_trade.close_ts)
            current_rates = data_point.get_values(name="highest_bid")
            profit = current_rates / opposite_trade.open_price - 1
            profit = profit * mask * self.scale_factor
        else:
            profit = np.zeros(len(timestamps))
        logger.debug("Get profit -> {0}".format(profit))

        return profit