class Context:
    def __init__(self):
        self.params = {}
        self.data_point = None

    def reset(self):
        self.params = {}
        self.data_point = None

    def set(self, name, value):
        self.params[name] = value

    def get(self, name):
        return self.params.get(name, 0)

    def set_dp(self, data_point):
        self.data_point = data_point

        ts = data_point.get_current_index()
        self.set("ts", ts)
        for key in ("lowest_ask", "highest_bid"):
            try:
                value = data_point.get_value(key)[0]
            except KeyError:
                value = 0
            finally:
                self.set(key, value)

        trade = self.params.get("trade")
        if trade is not None and trade.is_open:
            highest_bid = self.params.get("highest_bid")
            profit = trade.get_profit(highest_bid)
        else:
            profit = 0
        self.set("profit", profit)








