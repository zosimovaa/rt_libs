import sys

from .abstract_action import AbstractAction


class CorrectAction(AbstractAction):
    """The class describes the correct action applied by a neural network."""
    def __init__(self, ts, is_open, info=None):
        AbstractAction.__init__(self, ts, is_open, info=info)


class VoidAction(CorrectAction):
    """The class describes the correct action applied by a neural network."""
    def __init__(self, ts, is_open, info=None):
        CorrectAction.__init__(self, ts, is_open, info=info)


class FailAction(CorrectAction):
    """The class describes the correct action applied by a neural network."""
    def __init__(self, ts, is_open, info=None):
        CorrectAction.__init__(self, ts, is_open, info=info)


class TradeAction(CorrectAction):
    PROFIT_ACCURACY = 5
    """The class describes a trade operation"""

    def __init__(self, ts, price, market_fee=0.0, info=None):
        CorrectAction.__init__(self, ts, True, info=info)
        self.open_ts = ts
        self.open_price = price
        self.market_fee = market_fee

        self.close_ts = sys.maxsize
        self.close_price = None
        self.profit = 0

    def close(self, ts, price):
        """Method closes the trade and fixes profit"""
        if self.is_open:
            self.close_ts = ts
            self.close_price = price
            self.profit = self.get_profit(price)
            self.is_open = False

    def get_profit(self, price):
        """Method implements profit calculation """
        if self.is_open and self.open_price:
            return round(price / self.open_price - 1 - self.market_fee, self.PROFIT_ACCURACY)
        else:
            return self.profit
