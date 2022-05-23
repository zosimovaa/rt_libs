from ..core_actions import TradeAction


class TestMarketProvider:
    def __init__(self, context, market_fee):
        self.context = context
        self.market_fee = market_fee
        self.trade = None

    def open_trade(self):
        self.trade = TradeAction(self.context)
        self.context.update_trade(self.trade)
        return self.trade

    def close_trade(self):
        profit = self.get_profit()
        self.trade.close(self.context, profit)
        self.context.update_trade(self.trade)
        return self.trade

    def get_profit(self):
        current_price = self.context.get("highest_bid")
        if self.trade is not None and self.trade.is_open:
            profit = current_price / self.trade.open_price - 1 - self.market_fee
        else:
            profit = 0
        return profit

    def reset(self):
        self.trade = None
        self.context.update_trade(self.trade)



