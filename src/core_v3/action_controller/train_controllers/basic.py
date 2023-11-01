"""
Базовый тренировочный action controller
"""
import numpy as np

from ..action_router import Router4Action

from core_v2.actions import BadAction, VoidAction, TradeAction


class BasicTrainController(Router4Action):
    """Обработчик действий для обучения с 4 действиями"""

    def __init__(self, alias, market_fee=0, penalty=-1, wait_scale=0, open_scale=0, hold_scale=0, close_scale=1, last_points_mean=0):
        super().__init__(alias)

        self.market_fee = market_fee
        self.penalty = penalty

        self.wait_scale = wait_scale
        self.open_scale = open_scale
        self.hold_scale = hold_scale
        self.close_scale = close_scale

        self.last_points_mean = last_points_mean

        self.trade = None
        self.trade_opposite = None

        self.reset()

    def reset(self):
        super().reset()

        ts = self.context.get("ts")
        price = self.context.get("highest_bid")

        self.context.put("market_fee", self.market_fee)
        self.context.put("is_open", False)

        self.trade = TradeAction(ts, price)
        self.trade.close(ts, price)
        self.context.put("trade", self.trade)

        self.trade_opposite = TradeAction(ts, price)
        self.context.put("trade_opposite", self.trade_opposite)

    def apply_action_wait(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        else:
            reward = -1 * self._get_wh_reward() * self.wait_scale
            result_action = VoidAction(ts, is_open)

        return reward, result_action

    def apply_action_open(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        else:
            self._open_trade(ts)
            reward = -1 * self.trade_opposite.profit * self.open_scale
            result_action = self.trade

        return reward, result_action

    def apply_action_hold(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            reward = self._get_wh_reward() * self.hold_scale
            result_action = VoidAction(ts, is_open)
        else:
            reward = self.penalty
            result_action = BadAction(ts, is_open)

        return reward, result_action

    def apply_action_close(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            self._close_trade(ts)
            reward = self.trade.profit * self.close_scale
            result_action = self.trade
        else:
            reward = self.penalty
            result_action = BadAction(ts, is_open)

        return reward, result_action

    def _open_trade(self, ts):
        open_price = self.context.get("lowest_ask")
        self.trade = TradeAction(ts, open_price, market_fee=self.market_fee)
        self.context.put("trade", self.trade)

        close_price = self.context.get("highest_bid")
        self.trade_opposite.close(ts, close_price)
        self.context.put("is_open", True)

    def _close_trade(self, ts):
        close_price = self.context.get("highest_bid")
        self.trade.close(ts, close_price)

        open_price = self.context.get("highest_bid")
        self.trade_opposite = TradeAction(ts, open_price)
        self.context.put("trade_opposite", self.trade_opposite)
        self.context.put("is_open", False)

    def _get_wh_reward(self, name='highest_bid'):
        if self.last_points_mean > 0:
            data_point = self.context.get("data_point")
            current_price = data_point.get_value(name)
            values = data_point.get_values(name, num=self.last_points_mean + 1)
            values_diff = np.mean(np.diff(values)/current_price)
            return values_diff
        else:
            return 0

