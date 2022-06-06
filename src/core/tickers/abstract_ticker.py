import logging
import numpy as np

from ..actions import BadAction, TradeAction


logger = logging.getLogger(__name__)


class AbstractTickerOneFeature:
    """Класс реализует логику расчета награды/штрафа за действия и профита за торговые операции
    Используется в подготовительных сценариях
    """

    OPEN_PRICE_VALUE = 0.5

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
            action_result = BadAction(ts, self.handler.get(0))
        else:
            reward = self.reward
            action_result = None

        return reward, action_result

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(ts, self.handler.get(1))
        else:
            reward = self.reward
            self.trade = TradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            reward = self.reward
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, self.handler.get(2))
        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            feat_value = self.context.get("highest_bid")
            reward = feat_value
            self.trade.close()
            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, self.handler.get(3))
        return reward, action_result


class AbstractTickerOpenSignal(AbstractTickerOneFeature):
    def __init__(self, *args, **kwargs):
        AbstractTickerOneFeature.__init__(self, *args, **kwargs)

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(ts, self.handler.get(1))
        else:

            feat_value = self.context.get("highest_bid")
            self.context.set("open_feat_val", feat_value)

            reward = self.reward
            self.trade = TradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade

        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            feat_value = self.context.get("open_feat_val")
            reward = feat_value
            self.trade.close()
            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, self.handler.get(3))
        return reward, action_result


