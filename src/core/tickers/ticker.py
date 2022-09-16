import logging
import numpy as np

from ..actions import BadAction, TradeAction


logger = logging.getLogger(__name__)


class TickerBasic:
    """Класс реализует логику расчета награды/штрафа за действия и профита за торговые операции"""
    REWARD_SCALE_OPEN = 10
    REWARD_SCALE_CLOSE = 100
    NUM_MEAN_OBS = 2

    handler = {
        0: "_action_waiting",
        1: "_action_open_trade",
        2: "_action_hold",
        3: "_action_close_trade"
    }

    def __init__(self, context, penalty=-2, reward=0):
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
            self.trade = TradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade

            profit = self.trade.get_profit()
            reward = profit * self.REWARD_SCALE_OPEN
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
            profit = self.context.get("profit", domain="Trade")
            reward = profit * self.REWARD_SCALE_CLOSE
            self.trade.close()

            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result


class TickerExtendedReward(TickerBasic):
    """Класс реализует логику расчета награды/штрафа за действия и профита за торговые операции"""
    REWARD_SCALE_WAIT = 100
    REWARD_SCALE_OPEN = 10
    REWARD_SCALE_CLOSE = 100
    NUM_MEAN_OBS = 2

    handler = {
        0: "_action_waiting",
        1: "_action_open_trade",
        2: "_action_hold",
        3: "_action_close_trade"
    }

    def __init__(self, context, penalty=-2, reward=0):
        super().__init__(context, penalty=penalty, reward=reward)

    def _action_waiting(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            last_data_points_diff = self.get_last_diffs()
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = -rates_diff_mean / self.context.get("highest_bid") * self.REWARD_SCALE_WAIT

            action_result = None

        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            last_data_points_diff = self.get_last_diffs()
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = rates_diff_mean / self.context.get("highest_bid") * self.REWARD_SCALE_WAIT
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def get_last_diffs(self, column='lowest_ask'):
        data_point = self.context.data_point
        num = self.NUM_MEAN_OBS + 1
        feature = data_point.get_values(name=column, num=num, as_ndarray=False)
        return feature.diff().dropna().values


