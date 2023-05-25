"""
В модуле реализованы Ticker для расчета награды в сценариях торговли на абстрактных данных.
"""
import logging

from ..base_action_router import BaseActionRouter

from ...actions import BadAction, TradeAction


logger = logging.getLogger(__name__)


class AbstractSequencePrediction(BaseActionRouter):
    """
    Класс реализует логику расчета награды для сценариев обучения корректной последовательности и
    реакции на сигнал продажи.
    """

    def __init__(self, context, penalty=-1, reward=0):
        super().__init__(context=context, penalty=penalty, reward=reward)

    def _action_wait(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(ts, 0, is_open)
        else:
            reward = self.reward
            action_result = None

        return reward, action_result

    def _action_open(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(ts, 1, is_open)
        else:
            reward = self.reward
            self.trade = TradeAction(ts, 1, 0)
            self.context.set("trade", self.trade)
            self.context.set("is_open", True)
            action_result = self.trade

        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            reward = self.reward
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, 2, is_open)
        return reward, action_result

    def _action_close(self, ts, is_open):
        if is_open:
            self.trade.close(ts, 2)
            self.context.set("is_open", False)
            self.context.set("profit", self.trade.profit)

            reward = 1
            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, 3, is_open)
        return reward, action_result


class AbstractCloseSignal(AbstractSequencePrediction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _action_close(self, ts, is_open):
        if is_open:

            self.trade.close(ts, 2)
            self.context.set("is_open", False)
            dp = self.context.data_point
            signal_value = dp.get_value("close_signal")[0]

            if signal_value > 0:
                profit = 1
            else:
                profit = -1
            self.trade.profit = profit
            self.context.set("profit", profit)

            reward = profit
            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, 3, is_open)
        return reward, action_result


class AbstractTickerOpenSignal(AbstractSequencePrediction):
    """
    Класс реализует логику расчета награды для сценария обучения на сигнал покупки
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _action_open(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(ts, 1, is_open)
        else:
            dp = self.context.data_point
            open_signal_value = dp.get_value("open_signal")[0]
            self.context.set("open_signal_value", open_signal_value)

            reward = self.reward
            self.trade = TradeAction(ts, 1, 0)
            self.context.set("trade", self.trade)
            self.context.set("is_open", True)
            action_result = self.trade

        return reward, action_result

    def _action_close(self, ts, is_open):
        if is_open:
            self.trade.close(ts, 2)
            self.context.set("is_open", False)

            signal_value = self.context.get("open_signal_value")

            if signal_value > 0:
                profit = 1
            else:
                profit = -1
            self.trade.profit = profit
            self.context.set("profit", profit)

            reward = profit
            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, 3, is_open)
        return reward, action_result


class AbstractTickerCompleteTrade(AbstractSequencePrediction):
    """
    Класс реализует логику расчета награды для сценария обучения на сигналы покупки и продажи.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _action_open(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(ts, 1, is_open)
        else:
            dp = self.context.data_point
            open_signal_value = dp.get_value("open_signal")[0]
            self.context.set("open_signal_value", open_signal_value)

            reward = self.reward
            self.trade = TradeAction(ts, 1, 0)
            self.context.set("trade", self.trade)
            self.context.set("is_open", True)
            action_result = self.trade

        return reward, action_result

    def _action_close(self, ts, is_open):
        if is_open:
            self.trade.close(ts, 2)
            self.context.set("is_open", False)

            # -------------------------
            # Блок для 'прямого' датасета, где 1 - признак покупки/продажи

            dp = self.context.data_point
            close_signal = dp.get_value("close_signal")[0]
            open_signal = self.context.get("open_signal_value")

            if open_signal > 0 and close_signal > 0:
                profit = 1
            else:
               profit = -1
            # -------------------------

            self.trade.profit = profit
            self.context.set("profit", profit)

            reward = profit
            action_result = self.trade

        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result
