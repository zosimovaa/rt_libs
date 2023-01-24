import logging
import numpy as np

from .action_controller_interface import ActionControllerInterface
from ..actions import BadAction, TradeAction, OppositeTradeAction

logger = logging.getLogger(__name__)


class ActionControllerDiffReward(ActionControllerInterface):
    """Класс реализует логику расчета награды/штрафа за действия.
    Базовая версия - награда из профита выдается только при закрытии.

    OPEN - без награды (награда (профит) от OppositeTrade явно не улучшает ситуацию, надо исследовать)
    CLOSE - награда в виде профита
    В WAIT, HOLD - награда в виде изменения курса в пароцентах.
    Все веса регулируются коэффициентами, что позволяет какие-то факторы убирать в ноль или усиливать
    """
    handler = {
        0: "_action_wait",
        1: "_action_open",
        2: "_action_hold",
        3: "_action_close"
    }

    def __init__(self,
                 context,
                 penalty=-2, reward=0,
                 scale_wait=0, scale_open=0, scale_hold=0, scale_close=100,
                 num_mean_obs=2
                 ):
        self.context = context
        self.penalty = penalty
        self.reward = reward
        self.scale_wait = scale_wait
        self.scale_open = scale_open
        self.scale_hold = scale_hold
        self.scale_close = scale_close
        self.num_mean_obs = num_mean_obs

        self.trade = None
        self.opposite_trade = OppositeTradeAction(self.context)
        self.context.set("trade", self.opposite_trade, domain="OppositeTrade")

        logger.info("Initialized with penalty {0} and reward {1}.".format(penalty, reward))

    def reset(self):
        self.trade = None
        self.opposite_trade = OppositeTradeAction(self.context)
        self.context.set("trade", self.opposite_trade, domain="OppositeTrade")

    def apply_action(self, action):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open", domain="Trade")
        handler = getattr(self, self.handler[action])
        reward, action_result = handler(ts, is_open)
        return reward, action_result

    def _action_wait(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            reward = -self._get_diff_reward() * self.scale_wait
            action_result = None
        return reward, action_result

    def _action_open(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            self.trade = TradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade

            # Закрыть opposite_trade и рассчитать награду
            self.opposite_trade = self.context.get("trade", domain="OppositeTrade")
            profit = self.opposite_trade.get_profit()
            self.opposite_trade.close()
            reward = -profit * self.scale_open
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            reward = self._get_diff_reward() * self.scale_hold
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _action_close(self, ts, is_open):
        if is_open:
            profit = self.context.get("profit", domain="Trade")
            reward = profit * self.scale_close
            self.trade.close()
            action_result = self.trade

            self.opposite_trade = OppositeTradeAction(self.context)
            self.context.set("trade", self.opposite_trade, domain="OppositeTrade")
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _get_penalty(self, val=None):
        """Расчет штрафа. Если штрафне задан явно, то берем из базового значения"""
        value = self.penalty if val is None else val
        logger.debug("_get_penalty(): -> %s", value)
        return value

    def _get_diff_reward(self, name='highest_bid'):
        data_point = self.context.data_point
        values_diff = np.diff(data_point.get_values(name))
        value = data_point.get_value(name)
        values_rel = values_diff / value
        result = values_rel[-self.num_mean_obs:]
        return np.mean(result)
