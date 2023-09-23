"""
Классы для подключения к бирже полоник и обработке действий на ней.
Еще не реализовано
"""
import logging

from core_v1.actions import TradeAction


logger = logging.getLogger(__name__)


class PoloniexMarketProvider:
    def __init__(self, context, market_fee=0.00155):
        self.context = context
        self.market_fee = market_fee
        self.is_open = False

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

    def get_profit(self):
        pass

    def reset(self):
        self.trade = None




