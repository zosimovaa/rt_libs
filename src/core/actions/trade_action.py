import sys
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
        self.open_price = context.get("lowest_ask")
        self.market_fee = context.market_fee
        self.close_ts = sys.maxsize
        self.close_price = None
        self.is_open = True
        self.trade_volume = 0
        self.profit = self.get_profit()
        logger.debug("Trade {0} opened with {1} rate".format(self.id[-6:], self.open_price))

    def close(self):
        if self.is_open:
            self.close_ts = self.context.get("ts")
            self.close_price = self.context.get("highest_bid")
            self.profit = self.get_profit()
            self.is_open = False
            logger.debug("Trade {0} closed with {1} rate and profit {2}".format(
                self.id[-6:],
                self.close_price,
                self.profit))
        else:
            logger.debug("Trade {0} already closed".format(self.id[-6:]))

    def get_profit(self):
        if self.is_open:
            profit = round(self.context.get("highest_bid") / self.open_price - 1 - self.market_fee, 5)
        else:
            profit = 0.

        #logger.debug("Profit {0}. Trade status {1}".format(profit, self.is_open))
        return profit


class AbstractTradeAction(TradeAction):
    """Класс описывает торговую операцию для первого этапа обучения на абстрактных данных"""
    def __init__(self, context):
        TradeAction.__init__(self, context)

    def get_profit(self):
        if self.is_open:
            profit = self.context.get("highest_bid") - self.open_price
        else:
            profit = 0.

        #logger.debug("Profit {0}. Trade status {1}".format(profit, self.is_open))
        return profit
