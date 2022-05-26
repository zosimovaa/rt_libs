import uuid
import logging


logger = logging.getLogger(__name__)


class BaseAction:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.is_open = False


class BadAction(BaseAction):
    def __init__(self, context):
        BaseAction.__init__(self)
        self.ts = context.get("ts")
        self.action = context.get("action")


class TradeAction(BaseAction):
    def __init__(self, context):
        BaseAction.__init__(self)
        self.open_ts = context.get("ts")
        self.open_price = context.get("lowest_ask")
        self.close_ts = None
        self.close_price = None
        self.is_open = True
        self.profit = 0
        self.trade_volume = 0
        self.market_fee = context.market_fee

    def close(self, context, profit):
        self.is_open = False
        self.close_ts = context.get("ts")
        self.close_price = context.get("highest_bid")
        self.profit = profit