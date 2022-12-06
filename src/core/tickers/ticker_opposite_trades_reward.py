"""В модуле описана реализация TickerController с логикой расчета награды
для двух линий торговых операций"""

import logging
import numpy as np
from ..actions import BadAction, TradeAction, SimpleTradeAction


logger = logging.getLogger(__name__)


class TickerOppositeTradesReward:

    REWARD_WAIT = 10
    REWARD_OPEN = 10
    REWARD_HOLD = 10
    REWARD_CLOSE = 100
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
        self.opposite_trade = None

        self.penalty = penalty
        self.reward = reward

        logger.info("Initialized with penalty %s and reward %s.", penalty, reward)

    def reset(self):
        self.trade = None
        # Инициализация данных для расчета профита без торговой операции
        opposite_trade_open_price = self.context.get("highest_bid")
        ts = self.context.get("ts")
        self.opposite_trade = SimpleTradeAction(ts, opposite_trade_open_price)
        self.context.set("opposite_trade", self.opposite_trade, domain="OppositeTrade")
        logger.warning("Reset")

    def apply_action(self, action):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open", domain="Trade")
        handler = getattr(self, self.handler[action])
        reward, action_result = handler(ts, is_open)
        return reward, action_result

    def _get_penalty(self, val=None):
        """Расчет штрафа. Если штраф не задан явно, то берем из базового значения"""
        value = self.penalty if val is None else val
        logger.debug("_get_penalty(): -> {0}".format(value))
        return value

    def _action_waiting(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            last_data_points_diff = self.get_last_diffs()
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = -rates_diff_mean / self.context.get("highest_bid") * self.REWARD_WAIT
            action_result = None
        return reward, action_result

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            trade_open_price = self.context.get("lowest_ask")
            self.opposite_trade = SimpleTradeAction(ts, trade_open_price, market_fee=self.context.market_fee)
            self.context.set_trade(self.trade)
            action_result = self.trade

            # Инициализация данных для расчета профита без торговой операции
            opposite_trade_close_price = self.context.get("highest_bid")
            self.opposite_trade.close(ts, opposite_trade_close_price)

            reward = -self.opposite_trade.profit * self.REWARD_OPEN
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            last_data_points_diff = self.get_last_diffs()
            rates_diff_mean = np.mean(last_data_points_diff)
            reward = rates_diff_mean / self.context.get("highest_bid") * self.REWARD_HOLD
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            trade_close_price = self.context.get("highest_bid")
            self.trade.close(ts, trade_close_price)

            reward = self.trade.profit * self.REWARD_CLOSE
            action_result = self.trade

            # Инициализация данных для расчета профита без торговой операции
            opposite_trade_open_price = self.context.get("highest_bid")
            self.opposite_trade = SimpleTradeAction(ts, opposite_trade_open_price)
            self.context.set("opposite_trade", self.opposite_trade, domain="OppositeTrade")

        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def get_last_diffs(self, column='lowest_ask'):
        data_point = self.context.data_point
        num = self.NUM_MEAN_OBS + 1
        feature = data_point.get_values(name=column, num=num, as_ndarray=False)
        return feature.diff().dropna().values
