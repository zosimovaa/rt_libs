"""
В модуле реализованы Ticker для расчета награды в сценариях торговли на абстрактных данных.
"""
import logging

from ..actions import BadAction, AbstractTradeAction
from .action_controller_interface import ActionControllerInterface
from ..actions import BadAction, TradeAction, OppositeTradeAction


logger = logging.getLogger(__name__)


class AbstractTickerBasic(ActionControllerInterface):
    """
    Класс реализует логику расчета награды для сценариев обучения корректной последовательности и
    реакции на сигнал продажи.
    """

    handler = {
        0: "_action_waiting",
        1: "_action_open_trade",
        2: "_action_hold",
        3: "_action_close_trade"
    }

    def __init__(self, context, penalty=-1, reward=0):
        self.context = context

        self.trade = None

        self.penalty = penalty
        self.reward = reward
        logger.info("Initialized with penalty {0} and reward {1}.".format(penalty, reward))

    def reset(self):
        self.trade = None
        logger.warning("Reset")

    def apply_action(self, action):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open", domain="Trade")
        handler = getattr(self, self.handler[action])
        reward, action_result = handler(ts, is_open)
        return reward, action_result

    def _get_penalty(self, val=None):
        """Расчет штрафа. Если штрафне задан явно, то берем из базового значения"""
        value = self.penalty if val is None else val
        logger.debug("_get_penalty(): -> {0}".format(value))
        return value

    def _action_waiting(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            reward = self.reward
            action_result = None

        return reward, action_result

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            reward = self.reward
            self.trade = AbstractTradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            reward = self.reward
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            self.trade.close()

            signal_value = self.context.get("highest_bid")

            if signal_value > 0:
                self.trade.profit = 1
                self.context.set("profit", 1, domain="Trade")
            else:
                self.trade.profit = -1
                self.context.set("profit", -1, domain="Trade")

            reward = self.trade.profit
            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result


class AbstractTickerOpenSignal(AbstractTickerBasic):
    """
    Класс реализует логику расчета награды для сценария обучения на сигнал покупки
    """
    def __init__(self, *args, **kwargs):
        AbstractTickerBasic.__init__(self, *args, **kwargs)

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:

            open_signal_value = self.context.get("lowest_ask")
            self.context.set("open_signal_value", open_signal_value)

            reward = self.reward
            self.trade = AbstractTradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade

        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            self.trade.close()

            # -------------------------
            # Блок для 'прямого' датасета, где 1 - признак покупки
            open_signal = self.context.get("open_signal_value")
            if open_signal == 1:
                self.trade.profit = 1
                self.context.set("profit", 1, domain="Trade")
            else:
                self.trade.profit = -1
                self.context.set("profit", -1, domain="Trade")
            # -------------------------

            reward = self.trade.profit
            action_result = self.trade

        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result


class AbstractTickerCompleteTrade(AbstractTickerBasic):
    """
    Класс реализует логику расчета награды для сценария обучения на сигналы покупки и продажи.
    """
    def __init__(self, *args, **kwargs):
        AbstractTickerBasic.__init__(self, *args, **kwargs)

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:

            open_signal_value = self.context.get("lowest_ask")
            self.context.set("open_signal_value", open_signal_value)

            reward = self.reward
            self.trade = AbstractTradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade

        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            self.trade.close()

            # -------------------------
            # Блок для 'прямого' датасета, где 1 - признак покупки/продажи
            open_signal = self.context.get("open_signal_value")
            close_signal = self.context.get("highest_bid")

            if open_signal > 0 and close_signal > 0:
                self.trade.profit = 1
                self.context.set("profit", 1, domain="Trade")
            else:
                self.trade.profit = -1
                self.context.set("profit", -1, domain="Trade")
            # -------------------------

            reward = self.trade.profit
            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result
