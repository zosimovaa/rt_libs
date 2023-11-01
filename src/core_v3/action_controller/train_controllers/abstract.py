"""
Базовый контроллер со следующей логики формирования награды
 - wait: изменение курса между двумя точками со знаком минус (если растет без операции - это плохо)
 - open: профит от trade_opposite
 - hold: изменение курса между двумя точками (если растет c операции - это хорошо)
 - close: профит от trade

Уровень награды можно регулировать через коэффициенты xxxx_scale (в том числе и отключать)
"""

import numpy as np
from ...actions import BadAction, VoidAction, TradeAction
from .basic import BasicTrainController


class AbstractTrainControllerOpenSignal(BasicTrainController):
    """В open и close добавил работу с open_signal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.context.put("open_signal", 0)

    def apply_action_open(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        else:
            data_point = self.context.get("data_point")
            open_signal = data_point.get_value("open_signal")
            self.context.put("open_signal", open_signal)

            self._open_trade(ts)
            reward = -1 * self.trade_opposite.profit * self.open_scale
            result_action = self.trade
        return reward, result_action

    def apply_action_close(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            self._close_trade(ts)
            reward = self.trade.profit * self.close_scale
            result_action = self.trade
            self.context.put("open_signal", 0)
        else:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        return reward, result_action

class AbstractTrainControllerOpenCloseSignal(AbstractTrainControllerOpenSignal):
    """В open и close добавил работу с open_signal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.context.put("open_signal", 0)

    def apply_action_close(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:

            data_point = self.context.get("data_point")
            close_signal = data_point.get_value("close_signal")
            open_signal = self.context.get("open_signal")

            if close_signal == 1 and open_signal == 1:
                reward = 1
            else:
                reward = -1

            self._close_trade(ts)

            result_action = self.trade
            self.context.put("open_signal", 0)
        else:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        return reward, result_action
