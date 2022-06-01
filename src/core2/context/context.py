"""
Core context.
Хранит следующие сущности
 - data_point
 - trade
 - action
 - profit
 - reward

Извлекает следующие параметры и предоставляет к ним доступ:
 - ts
 - lowest_ask
 - highest_bid
 - is_open
"""


class Context:
    def __init__(self, market_fee):
        self.data_point = None
        self.trade = None
        self.params = dict()

    def reset(self):
        self.data_point = None
        self.trade = None
        self.params = dict()

    def set(self, param, value):
        self.params[param] = value

    def get(self, param, default=None):
        return self.params.get(param, default)

    def update_datapoint(self, data_point):
        self.data_point = data_point

        self.set("ts", data_point.get_current_ts())
        self.set("lowest_ask", data_point.get_value("lowest_ask"))
        highest_bid = data_point.get_value("highest_bid")
        self.set("highest_bid", highest_bid)

        if self.trade is not None:
            self.trade.update(highest_bid)
            self.set("is_open", self.trade.is_open)
            self.set("profit", self.trade.profit)
        else:
            self.set("is_open", False)
            self.set("profit", 0)

    def update_trade(self, trade):
        self.trade = trade
