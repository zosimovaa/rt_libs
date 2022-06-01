import logging

from ..actions import TradeAction


logger = logging.getLogger(__name__)


class TestMarketProvider:
    def __init__(self, context):
        self.context = context
        self.trade = None

    def open_trade(self):
        ts = self.context.get("ts", 0)
        lowest_ask = self.context.get("lowest_ask", 0)
        highest_bid = self.context.get("highest_bid", 0)
        market_fee = self.context.get("market_fee", 0)

        self.trade = TradeAction(ts, lowest_ask, highest_bid, market_fee)
        self.context.update_trade(self.trade)
        return self.trade

    def close_trade(self):
        ts = self.context.get("ts", 0)
        highest_bid = self.context.get("highest_bid", 0)
        self.trade.close(ts, highest_bid)

        return self.trade

    def reset(self):
        self.trade = None




