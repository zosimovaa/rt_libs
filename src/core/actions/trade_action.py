import sys
import logging
from .base_action import BaseAction


logger = logging.getLogger(__name__)


class TradeAction(BaseAction):
    """The class describes a trade operation"""
    def __init__(self, context):
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
        logger.debug("Trade %s opened with %s rate", self.id[-6:], self.open_price)

    def close(self):
        """Method closes the trade and fix profit"""
        if self.is_open:
            self.close_ts = self.context.get("ts")
            self.close_price = self.context.get("highest_bid")
            self.profit = self.get_profit()
            self.is_open = False
            logger.debug("Trade %s closed with %s rate and profit %s", self.id[-6:], self.close_price, self.profit)
        else:
            logger.debug("Trade %s already closed", self.id[-6:])

    def get_profit(self):
        """Method implements profit calculation """
        if self.is_open:
            profit = round(self.context.get("highest_bid") / self.open_price - 1 - self.market_fee, 5)
        else:
            profit = 0.

        #logger.debug("Profit {0}. Trade status {1}".format(profit, self.is_open))
        return profit


class AbstractTradeAction(TradeAction):
    """The class describes a trading operation for the first stage of learning on abstract data"""
    def __init__(self, context):
        TradeAction.__init__(self, context)

    def get_profit(self):
        if self.is_open:
            profit = self.context.get("highest_bid") - self.open_price
        else:
            profit = 0.

        #logger.debug("Profit {0}. Trade status {1}".format(profit, self.is_open))
        return profit


class SimpleTradeAction(BaseAction):
    """The class describes a trade operation. Implementation without context
    All parameters should be passed into methods
    """
    def __init__(self, ts, price, market_fee=0):
        BaseAction.__init__(self)
        self.open_ts = ts
        self.open_price = price
        self.market_fee = market_fee

        self.close_ts = sys.maxsize
        self.close_price = None

        self.profit = 0

        self.is_open = True
        logger.debug('Trade %s opened with %s rate', self.id[-6:], self.open_price)

    def close(self, ts, price):
        """Method closes the trade and fix profit"""
        if self.is_open:
            self.close_ts = ts
            self.close_price = price
            self.profit = self.get_profit(price)
            self.is_open = False
            logger.debug("Trade %s closed with %s rate and profit %s", self.id[-6:], self.close_price, self.profit)
        else:
            logger.debug("Trade %s already closed", self.id[-6:])

    def get_profit(self, price):
        """Method implements profit calculation """
        if self.is_open:
            profit = round(price / self.open_price - 1 - self.market_fee, 5)
        else:
            profit = 0.
        return profit
