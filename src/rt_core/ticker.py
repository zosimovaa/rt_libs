import logging
import numpy as np

from .core_actions import BadAction


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class Ticker:
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

    def __init__(self, context, trade_controller, penalty=-2, reward=0):
        self.context = context
        self.trade_controller = trade_controller
        self.penalty = penalty
        self.reward = reward
        logger.info("Initialized with penalty {0} and reward {1}.".format(penalty, reward))

    def reset(self):
        self.trade_controller.reset()
        logger.warning("Reset")

    def apply_action(self, action):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open", domain="Trade")
        profit = self.trade_controller.get_profit()
        self.context.set("profit", profit, domain="Trade")

        handler = getattr(self, self.handler[action])

        reward, action_result = handler(ts, is_open)

        self.context.set("reward", reward)
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
            last_data_points_diff = self.context.data_point.get_last_diffs(self.NUM_MEAN_OBS)
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = -rates_diff_mean / self.context.get("highest_bid") * self.REWARD_SCALE_WAIT
            action_result = None

        return reward, action_result

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            action_result = self.trade_controller.open_trade()
            profit = self.trade_controller.get_profit()
            reward = profit * self.REWARD_SCALE_OPEN
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            last_data_points_diff = self.context.data_point.get_last_diffs(self.NUM_MEAN_OBS)
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = rates_diff_mean / self.context.get("highest_bid") * self.REWARD_SCALE_WAIT
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            profit = self.trade_controller.get_profit()
            reward = profit * self.REWARD_SCALE_CLOSE
            action_result = self.trade_controller.close_trade()
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result
