from .base_action import BaseAction


class TradeAction(BaseAction):
    """Класс описывает торговую операцию"""
    def __init__(self, ts, lowest_ask, highest_bid, market_fee):
        """Инстанс класса создается при открытии торговой операции"""
        BaseAction.__init__(self)
        self.open_ts = ts
        self.open_price = lowest_ask
        self.market_fee = market_fee
        self.close_ts = None
        self.close_price = None
        self.is_open = True
        self.last_price = highest_bid
        self.trade_volume = 0
        self.profit = self.get_profit()

    def update(self, highest_bid):
        if self.is_open:
            self.last_price = highest_bid
            self.profit = self.get_profit()

    def close(self, ts, highest_bid):
        if self.is_open:
            self.close_ts = ts
            self.close_price = highest_bid
            self.last_price = highest_bid
            self.profit = self.get_profit()
            self.is_open = False

    def get_profit(self):
        if self.is_open:
            profit = round(self.last_price / self.open_price - 1 - self.market_fee, 5)
        else:
            profit = 0.
        return profit
