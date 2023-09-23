import sys

from .base_action import BaseAction


class TradeAction(BaseAction):
    """The class describes a trade operation"""
    def __init__(self, open_ts, open_price, market_fee):
        BaseAction.__init__(self)

        self.open_ts = open_ts
        self.open_price = open_price
        self.market_fee = market_fee
        self.is_open = True

        self.close_ts = sys.maxsize
        self.close_price = None
        self.profit = 0

    def close(self, close_ts, close_price):
        """Method closes the trade and fixes profit"""
        if self.is_open:
            self.close_ts = close_ts
            self.close_price = close_price
            self.profit = self.get_profit(close_price)
            self.is_open = False
        else:
            pass

    def get_profit(self, price):
        """Method implements profit calculation """
        return round(price / self.open_price - 1 - self.market_fee, 5)
