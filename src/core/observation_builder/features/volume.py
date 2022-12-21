"""
Trade volumes features
"""
import logging

from .abstract_class import AbstractFeature

logger = logging.getLogger(__name__)


class TradeBalanceFeature(AbstractFeature):
    DELTA = 0.0000000000001  # Option to avoid division by 0.

    def __init__(self, context):
        super().__init__(context)

    def _get(self):
        dp = self.context.data_point
        buy_vol = dp.get_values("buy_vol")
        sell_vol = dp.get_values("sell_vol")
        trade_balance = (buy_vol - sell_vol) / (buy_vol + sell_vol + self.DELTA)
        return trade_balance
