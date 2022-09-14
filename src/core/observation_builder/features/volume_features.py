from .abstract_feature import AbstractFeature
import logging
import numpy as np


logger = logging.getLogger(__name__)


class TradeBalanceFeature(AbstractFeature):
    DELTA = 0.0000000000001 # Параметр, чтобы избежать деления на 0.

    def __init__(self, context):
        super().__init__(context)

    def get(self):
        dp = self.context.data_point
        buy_vol = dp.get_values("buy_vol").values
        sell_vol = dp.get_values("sell_vol").values
        trade_balance = (buy_vol - sell_vol) / (buy_vol + sell_vol + self.DELTA)
        return trade_balance
