import logging

logger = logging.getLogger(__name__)


class Context:
    def __init__(self, market_fee=0):
        self.data_point = None
        self.market_fee = market_fee
        self.params = dict()

    def reset(self):
        self.data_point = None
        self.params = dict()

    def set(self, param, value):
        self.params[param] = value

    def get(self, param, default=None):
        return self.params.get(param, default)

    def update_datapoint(self, data_point):
        self.data_point = data_point

        ts = data_point.get_current_ts()
        self.set("ts", ts)

        lowest_ask = data_point.get_price("lowest_ask")
        self.set("lowest_ask", lowest_ask)

        highest_bid = data_point.get_price("highest_bid")
        self.set("highest_bid", highest_bid)

    def update_trade(self, trade):
        for key in trade.__dict__:
            self.set(key, trade.__dict__[key])

    def get_profit(self):
        if self.params.get("is_open", False):
            open_price = self.params.get("open_price")
            current_price = self.params.get("highest_bid")
            profit = current_price / open_price - 1 - self.market_fee
        else:
            profit = 0
        return profit


class ContextWithDomains:
    DEFAULT_DOMAIN = "Common"

    def __init__(self, market_fee=0):
        self.market_fee = market_fee
        self.data_point = None
        self.trade = None
        self.params = {self.DEFAULT_DOMAIN: dict()}

    def reset(self):
        self.data_point = None
        self.trade = None
        self.params = dict()

    def set(self, param, value, domain=DEFAULT_DOMAIN):
        if domain not in self.params:
            self.params[domain] = dict()
        self.params[domain][param] = value

    def get(self, param, default=0, domain=DEFAULT_DOMAIN):
        return self.params.get(domain, dict()).get(param, default)

    def update_datapoint(self, data_point):
        if data_point is not None:
            self.data_point = data_point

            ts = data_point.get_current_ts()
            self.set("ts", ts)

            lowest_ask = data_point.get_price("lowest_ask")
            self.set("lowest_ask", lowest_ask)

            highest_bid = data_point.get_price("highest_bid")
            self.set("highest_bid", highest_bid)

    def update_trade(self, trade):
        if trade is not None:
            self.set("is_open_prev", trade.__dict__.get("is_open", False), domain="Trade")
            for key in trade.__dict__:
                self.set(key, trade.__dict__[key], domain="Trade")
        else:
            self.set("is_open", False, domain="Trade")
