"""
Профит контроллер с расчетоv профита в о;идании в зависимости от текущего уровня награды -
Если профит положительный - даем награду как есть (на основе разницы курса в предыдущей точкой)
Если профит отрицательный - дает только отрицательную награду, а положительную обнуляем.
Награда за закрытие и открытие на основе торговой операции и регулируется через scale.
"""
import logging
import numpy as np

from ..base_action_router import BaseActionRouter
from ...actions import BadAction, TradeAction

logger = logging.getLogger(__name__)


class ActionControllerProfitDependency(BaseActionRouter):
    """Класс реализует логику расчета награды/штрафа за действия.
    Базовая версия -

    OPEN - без награды (награда (профит) от OppositeTrade явно не улучшает ситуацию, надо исследовать)
    CLOSE - награда в виде профита
    В WAIT, HOLD - награда в виде изменения курса в процентах.
    Все веса регулируются коэффициентами, что позволяет какие-то факторы убирать в ноль или усиливать
    """

    def __init__(self,
                 context,
                 penalty=-2, reward=0, market_fee=0.0015,
                 scale_wait=0, scale_open=0, scale_hold=0, scale_close=100,
                 num_mean_obs=2):

        super().__init__(context=context, penalty=penalty, reward=reward)

        self.market_fee = market_fee

        self.scale_wait = scale_wait
        self.scale_open = scale_open
        self.scale_hold = scale_hold
        self.scale_close = scale_close

        self.num_mean_obs = num_mean_obs

        self.opposite_trade = None

        self.reset()

    def reset(self):
        """Reset current action controller state"""
        self.trade = None
        self.context.set("trade", self.trade)
        self.context.set("is_open", False)
        self.context.set("market_fee", self.market_fee)

        ts = self.context.get("ts")
        highest_bid = self.context.get("highest_bid")
        self.opposite_trade = TradeAction(ts, highest_bid, self.market_fee)
        self.context.set("opposite_trade", self.opposite_trade)

    def apply_action_wait(self):
        is_open = self.context.get("is_open")
        ts = self.context.get("ts")

        if is_open:
            # Wrong action, penalty
            reward = self._get_penalty()
            action_result = BadAction(ts, 0, is_open)
        else:
            # так быстрее
            if self.scale_wait:
                # минуc добавляется т.к. при росте курса в отсутствии открытой операции нужно дать штраф.
                reward = -self._get_diff_reward() * self.scale_wait
            else:
                reward = 0
            action_result = None
        return reward, action_result

    def apply_action_open(self):
        is_open = self.context.get("is_open")
        ts = self.context.get("ts")

        if is_open:
            # Wrong action, penalty
            reward = self._get_penalty()
            action_result = BadAction(ts, 1, is_open)
        else:
            # Close opposite trade
            highest_bid = self.context.get("highest_bid")
            self.opposite_trade.close(ts, highest_bid)

            # Open trade
            open_price = self.context.get("lowest_ask")
            self.trade = TradeAction(ts, open_price, self.market_fee)
            self.context.set("trade", self.trade)
            self.context.set("is_open", True)
            action_result = self.trade

            # Calculate reward
            reward = -self.opposite_trade.profit * self.scale_open
        return reward, action_result

    def apply_action_hold(self):
        is_open = self.context.get("is_open")
        ts = self.context.get("ts")

        if is_open:


            if self.scale_hold:
                highest_bid = self.context.get("highest_bid")
                profit = self.trade.get_profit(highest_bid)

                if profit >= 0:
                    reward = self._get_diff_reward() * self.scale_hold
                else:
                    reward = min(0, self._get_diff_reward() * self.scale_hold)

            else:
                reward = 0
            action_result = None



        else:
            # Wrong action, penalty
            reward = self._get_penalty()
            action_result = BadAction(ts, 2, is_open)
        return reward, action_result

    def apply_action_close(self):
        is_open = self.context.get("is_open")
        ts = self.context.get("ts")

        if is_open:
            highest_bid = self.context.get("highest_bid")
            profit = self.trade.get_profit(highest_bid)
            reward = profit * self.scale_close
            self.trade.close(ts, highest_bid)
            action_result = self.trade
            self.context.set("is_open", False)
            self.opposite_trade = TradeAction(ts, highest_bid, self.market_fee)
        else:
            # Wrong action, penalty
            reward = self._get_penalty()
            action_result = BadAction(ts, 3, is_open)
        return reward, action_result

    def _get_penalty(self, val=None):
        """Расчет штрафа. Если штрафне задан явно, то берем из базового значения"""
        value = self.penalty if val is None else val
        return value

    def _get_diff_reward(self, name='highest_bid'):
        data_point = self.context.data_point
        values_diff = np.diff(data_point.get_values(name))
        value_norm = data_point.get_value(name)[0]
        values_rel = values_diff / value_norm
        result = np.mean(values_rel[-self.num_mean_obs:])
        return result
