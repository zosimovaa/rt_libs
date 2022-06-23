import logging
import numpy as np

from ..actions import BadAction, TradeAction


logger = logging.getLogger(__name__)


class Ticker_:
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

    def __init__(self, context, penalty=-2, reward=0, market_fee=0.0015):
        self.context = context

        self.trade = None

        self.market_fee = market_fee
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
            last_data_points_diff = self.context.data_point.get_last_diffs(self.NUM_MEAN_OBS)
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = -rates_diff_mean / self.context.get("highest_bid") * self.REWARD_SCALE_WAIT
            action_result = None

        return reward, action_result

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(ts, self.handler.get(1))
        else:

            # >>> trade controller logic start
            lowest_ask = self.context.get("lowest_ask", 0)
            highest_bid = self.context.get("highest_bid", 0)

            action_result = TradeAction(self.context)
            self.trade = action_result
            self.context.set_trade(self.trade)
            # <<< trade controller logic end

            profit = self.trade.get_profit()
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
            action_result = BadAction(ts, self.handler.get(2))
        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            profit = self.context.get("profit", domain="Trade")
            reward = profit * self.REWARD_SCALE_CLOSE

            # >>> trade controller logic start
            highest_bid = self.context.get("highest_bid", 0)
            self.trade.close(ts, highest_bid)
            # <<< trade controller logic end

            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(ts, self.handler.get(3))
        return reward, action_result
