from .abstract_feature import AbstractFeature
import logging
import numpy as np


logger = logging.getLogger(__name__)


class TradeStateFeature(AbstractFeature):
    """Достает из контекста текущее состояние торговой операции"""
    def __init__(self, context):
        super().__init__(context)

    def get(self):
        trade_state = self.context.get("is_open", domain="Trade")
        logger.debug("Get trade state() -> {}".format(trade_state))
        return trade_state


class Rates1DFeature(AbstractFeature):
    SCALE_FACTOR = 10

    def __init__(self, context):
        super().__init__(context)

    def get(self):
        data_point = self.context.data_point
        current_price = self.context.get("highest_bid")
        rates = (data_point.get_values("highest_bid").values / current_price - 1) * self.SCALE_FACTOR
        logger.debug("Get rates -> current price {0} | {1}".format(current_price, rates))
        return rates


class ProfitFeature(AbstractFeature):
    SCALE_FACTOR = 10

    def __init__(self, context):
        super().__init__(context)

    def get(self):
        timestamps = self.context.data_point.get_timestamps()
        data_point = self.context.data_point
        trade = self.context.trade

        trade_state = self.context.get("is_open", domain="Trade")

        if trade_state:
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
            current_rates = data_point.get_values("highest_bid").values
            profit = current_rates / trade.open_price - 1 - self.context.market_fee
            profit = profit * mask * self.SCALE_FACTOR
        else:
            profit = np.zeros(len(timestamps))
        logger.debug("Get profit -> {0}".format(profit))

        return profit
