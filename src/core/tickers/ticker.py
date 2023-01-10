import logging
import numpy as np

from ..actions import BadAction, TradeAction


logger = logging.getLogger(__name__)


class TickerBasic:
    """Класс реализует логику расчета награды/штрафа за действия.
    Базовая версия - награда из профита выдается только при открытии и закрытии. В ожидании будет награда только в виде
    штрафа за неправильные действия.
    """

    handler = {
        0: "_action_wait",
        1: "_action_open",
        2: "_action_hold",
        3: "_action_close"
    }

    def __init__(self, context, penalty=-2, reward=0,
                 scale_wait=10, scale_open= 10, scale_hold= 10, scale_close=100, num_mean_obs=2):
        self.context = context
        self.penalty = penalty
        self.reward = reward
        self.scale_wait = scale_wait
        self.scale_open = scale_open
        self.scale_hold = scale_hold
        self.scale_close = scale_close
        self.num_mean_obs = num_mean_obs

        self.trade = None
        logger.info("Initialized with penalty {0} and reward {1}.".format(penalty, reward))

    def reset(self):
        self.trade = None

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

    def _action_wait(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            reward = self.reward * self.scale_wait
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

            profit = self.trade.get_profit()
            reward = profit * self.scale_open
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            reward = self.reward * self.scale_hold
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
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def get_last_diffs(self, column='lowest_ask'):
        data_point = self.context.data_point
        num = self.num_mean_obs + 1
        feature_values = data_point.get_values(column, num=num)
        result = np.diff(feature_values)
        return result


class TickerExtendedReward(TickerBasic):
    """Класс реализует логику расчета награды/штрафа за действия и профита за торговые операции"
    Помимо награзы в виде профита за открытие/закрытие добавляется награда в ожидании в виде изменения курса.
    """

    handler = {
        0: "_action_wait",
        1: "_action_open",
        2: "_action_hold",
        3: "_action_close"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _action_wait(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            last_data_points_diff = self.get_last_diffs()
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = -rates_diff_mean / self.context.get("highest_bid") * self.scale_wait
            action_result = None
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            last_data_points_diff = self.get_last_diffs()
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = rates_diff_mean / self.context.get("highest_bid") * self.scale_hold
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result


class TickerExtendedReward2(TickerExtendedReward):
    """На холде будет строить награду из профита"""

    def _action_hold(self, ts, is_open):
        if is_open:
            profit = self.context.get("profit", domain="Trade")
            reward = profit * self.scale_hold
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result
