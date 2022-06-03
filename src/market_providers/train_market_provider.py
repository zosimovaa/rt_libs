import logging

from core.actions import TradeAction


logger = logging.getLogger(__name__)


class TestMarketProvider:
    def __init__(self, context, market_fee=0.00155):
        self.context = context
        self.trade = None
        self.market_fee = market_fee

    def open_trade(self):
        ts = self.context.get("ts", 0)
        lowest_ask = self.context.get("lowest_ask", 0)
        highest_bid = self.context.get("highest_bid", 0)

        self.trade = TradeAction(ts, lowest_ask, highest_bid, self.market_fee)
        self.context.set_trade(self.trade)
        return self.trade

    def close_trade(self):
        ts = self.context.get("ts", 0)
        highest_bid = self.context.get("highest_bid", 0)
        self.trade.close(ts, highest_bid)

        return self.trade

    def reset(self):
        self.trade = None




