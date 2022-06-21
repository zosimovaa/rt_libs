"""
Core context.
"""
import logging

from .basic_context import BasicContext

logger = logging.getLogger(__name__)


class AbstractContextSequencePrediction(BasicContext):
    LOWEST_ASK = 0

    def __init__(self, *args, **kwargs):
        BasicContext.__init__(self, *args, **kwargs)

    def update_datapoint(self, data_point):
        self.data_point = data_point
        self.set("data_point", self.data_point.data, domain="Data")

        # update price params in context
        self.set("ts", data_point.get_current_ts())
        self.set("lowest_ask", self.LOWEST_ASK)
        self.set("highest_bid", data_point.get_value("feature"))

        # update trade with new rates
        self.update_trade()


class AbstractContextCloseSignal(BasicContext):
    LOWEST_ASK = 0

    def __init__(self, *args, **kwargs):
        BasicContext.__init__(self, *args, **kwargs)

    def update_datapoint(self, data_point):
        self.data_point = data_point
        self.set("data_point", self.data_point.data, domain="Data")

        # update price params in context
        self.set("ts", data_point.get_current_ts())
        self.set("lowest_ask", data_point.get_value("close_signal"))
        self.set("highest_bid", data_point.get_value("close_signal"))

        # update trade with new rates
        self.update_trade()


class AbstractContextOpenSignal(BasicContext):
    def __init__(self, *args, **kwargs):
        BasicContext.__init__(self, *args, **kwargs)

    def update_datapoint(self, data_point):
        self.data_point = data_point
        self.set("data_point", self.data_point.data, domain="Data")

        # update price params in context
        self.set("ts", data_point.get_current_ts())
        self.set("lowest_ask", data_point.get_value("open_signal"))
        self.set("highest_bid", data_point.get_value("open_signal"))

        # update trade with new rates
        self.update_trade()


class AbstractContextCompleteTrade(BasicContext):
    def __init__(self, *args, **kwargs):
        BasicContext.__init__(self, *args, **kwargs)

    def update_datapoint(self, data_point):
        self.data_point = data_point
        self.set("data_point", self.data_point.data, domain="Data")

        # update price params in context
        self.set("ts", data_point.get_current_ts())
        self.set("lowest_ask", data_point.get_value("open_signal"))
        self.set("highest_bid", data_point.get_value("close_signal"))

        # update trade with new rates
        self.update_trade()
