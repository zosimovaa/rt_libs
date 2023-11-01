import numpy as np
from .basic import BasicTrainController
from core_v2.actions import BadAction, VoidAction, TradeAction

class TrainControllerWHProfit(BasicTrainController):
    """Для WAIT и HOLD при отрицательном профите будем давать только отрицательную награду, пири положительно - только положительную"""

    def __init__(self, *args, **kwargs):
        self.edge = kwargs.pop("edge", 0)
        super().__init__(*args, **kwargs)

    def apply_action_wait(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        else:
            result_action = VoidAction(ts, is_open)
            if self.wait_scale:

                price = self.context.get("highest_bid")
                profit_opposite = self.trade_opposite.get_profit(price)

                reward = -1 * profit_opposite * self.wait_scale

            else:
                reward = 0

        return reward, result_action

    def apply_action_hold(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")

        if is_open:
            result_action = VoidAction(ts, is_open)
            if self.hold_scale:

                price = self.context.get("highest_bid")
                profit = self.trade.get_profit(price)

                reward = profit * self.hold_scale

            else:
                reward = 0
        else:
            reward = self.penalty
            result_action = BadAction(ts, is_open)

        return reward, result_action