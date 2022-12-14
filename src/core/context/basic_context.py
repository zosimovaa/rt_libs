"""
Core context.

Доступ к параметрам через get более безопасен -> тогда вариант с доменами выглядит ОК
Домены
 - Data - хранит data_point
 - action
 - Trade - все связанное с торговой операцией

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

import logging


logger = logging.getLogger(__name__)


class BasicContext:
    DEFAULT_DOMAIN = "Common"

    def __init__(self, market_fee=0):
        self.market_fee = market_fee
        self.data_point = None
        self.trade = None
        self.params = {self.DEFAULT_DOMAIN: dict()}
        #logger.debug("Instance created with market_fee {}".format(market_fee))

    def reset(self):
        self.data_point = None
        self.trade = None
        self.params = dict()

    def set(self, param, value, domain=DEFAULT_DOMAIN):
        if domain not in self.params:
            self.params[domain] = dict()
        self.params[domain][param] = value
        #logger.debug("Value {0} set to param {1}".format(value, param))

    def get(self, param, default=0, domain=DEFAULT_DOMAIN):
        return self.params.get(domain, dict()).get(param, default)

    def update_datapoint(self, data_point):
        self.data_point = data_point
        # update price params in context
        self.set("ts", data_point.get_current_index())
        self.set("lowest_ask", data_point.get_value("lowest_ask"))
        self.set("highest_bid", data_point.get_value("highest_bid"))
        # update trade status if exists
        self.update_trade()

    # todo перенести в action_controller или переделать на event bus
    def update_trade(self):
        if self.trade is not None:
            self.set("is_open_prev", self.trade.is_open, domain="Trade")
            self.set("is_open", self.trade.is_open, domain="Trade")
            self.set("profit", self.trade.get_profit(), domain="Trade")
            #for key in self.trade.__dict__:
                #self.set(key, self.trade.__dict__[key], domain="Trade")

        else:
            self.set("is_open_prev", False, domain="Trade")
            self.set("is_open", False, domain="Trade")
            self.set("profit", 0, domain="Trade")

    # todo перенести в action_controller или переделать на event bus
    def set_trade(self, trade):
        self.trade = trade
        self.update_trade()
