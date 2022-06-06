import logging
from .base_action import BaseAction


logger = logging.getLogger(__name__)


class TradeAction(BaseAction):
    """Класс описывает торговую операцию"""
    def __init__(self, context):
        """Инстанс класса создается при открытии торговой операции"""
        BaseAction.__init__(self)
        self.context = context
        self.open_ts = context.get("ts")
        self.open_price = context.get("lowest_ask") + 0.5
        self.market_fee = context.market_fee
        self.close_ts = None
        self.close_price = None
        self.is_open = True
        self.trade_volume = 0
        self.profit = self.get_profit()
        logger.debug("Trade {0} opened as {1} with {2} rate".format(self.open_ts, self.id, self.open_price))

    def close(self):
        if self.is_open:
            self.close_ts = self.context.get("ts")
            self.close_price = self.context.get("highest_bid")
            self.profit = self.get_profit()
            self.is_open = False
        logger.debug("Trade {0} closed as {1} with {2} rate and profit {3}".format(
            self.close_ts,
            self.id,
            self.close_price,
            self.profit))

    def get_profit(self):
        if self.is_open:
            profit = round(self.context.get("highest_bid") / self.open_price - 1 - self.market_fee, 5)
        else:
            profit = 0.
        return profit
