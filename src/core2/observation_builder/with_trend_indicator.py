"""
Базовый билдер с двумя фичами:
 - состояние сделки
 - данные для НС
   - курс
   - профит

Данные для сети подаются в формате [num_of_points, num_features]
        rate perp        profit repr
array([[-1.9651896e-01,  0.0000000e+00],
       [-1.8928000e-01,  0.0000000e+00],
       [-1.8809754e-01,  0.0000000e+00]]
"""

import logging
import numpy as np
from collections import deque

from .interface import ObservationBuilderInterface

logger = logging.getLogger(__name__)


class ObservationBuilderFutureFeature(ObservationBuilderInterface):
    """Билдер с 2-мя фичами. Без кэша.
    Работает дольше, чем с кэшом - на обучении скорость падает в 3 раза.
    Более стабильный вариант (нет проблемы с инвалидацией кэша), подходит для торговли."""
    def __init__(self, context):
        self.context = context

    def reset(self):
        pass

    def get(self, data_point):
        # trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # rates representation
        current_price = self.context.get("highest_bid")
        rates = (data_point.get_prices("highest_bid").values / current_price - 1) * 10

        # profit representation
        profit = self._get_profit(data_point, trade_state)

        # trend indicator representation
        trend = self._get_trend()

        # observation
        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            trend.reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array([trade_state], dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

    def _get_profit(self, data_point, trade_state):
        if trade_state:
            profit = data_point.get_prices("highest_bid").values.copy().reshape(-1)
            mask = data_point.get_timestamps() > self.context.get("open_ts", domain="Trade")
            profit = profit / self.context.get("open_price", domain="Trade") - 1 - self.context.market_fee
            profit = profit * mask * 10
        else:
            profit = np.zeros(data_point.offset)
        return profit

    def _get_trend(self):
        ti = []
        # todo - потери времени на цикле.
        for i in range(self.context.data_point.offset):
            current_value = self.context.data_point.get_price("highest_bid", cursor=i)
            future_values = self.context.data_point.get_future_prices("highest_bid", cursor=i).values.reshape(-1)

            diff = np.array(future_values / current_value - 1) * 100
            coeffs = np.linspace(1.0, 0.5, len(diff))

            if len(coeffs):
                trend_indicator = np.average(diff, weights=coeffs)
            else:
                trend_indicator = 0
            ti.append(trend_indicator)
        return np.array(ti)


class ObservationBuilderFutureFeatureCache(ObservationBuilderInterface):
    """Билдер с добавлением признака тренда. С кэшом. Требует инициализации (сброс с датапоинтом в контексте)."""

    def __init__(self, context):
        self.context = context
        self.history = None  # Кэш истории
        self.last_update = 0  # Контроль апдейта кэша

    def reset(self):
        self.history = None
        self.last_update = 0

    def init_history(self):
        idxs = self.context.data_point.get_timestamps()
        self.history = deque(maxlen=len(idxs))

        for idx in idxs:
            price = self.context.data_point.get_price("highest_bid", cursor=idx)
            future_values = self.context.data_point.get_future_prices("highest_bid", cursor=idx).values.reshape(-1)
            trend = self._get_trend(future_values, price)
            profit = 0

            obs_point = [price, profit, trend]
            self.history.append(obs_point)

    def get(self, data_point):
        if self.history is None:
            self.init_history()

        self._build_history_point()

        history = np.array(self.history)

        rates = history[:, 0]
        profits = history[:, 1]
        trends = history[:, 2]

        current_price = self.context.get("highest_bid")
        rates = (rates / current_price - 1) * 10

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profits.reshape(-1, 1),
            trends.reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array([self.context.get("is_open", domain="Trade")], dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

    def _build_history_point(self):
        ts = self.context.get("ts", default=-1)
        if self.last_update != ts:

            # trade state feature
            trade_state = self.context.get("is_open", domain="Trade")

            # 1. Построить текущую точку
            price = self.context.get("highest_bid")
            profit = self._get_profit()

            future_prices = self.context.data_point.get_future_prices("highest_bid")
            future_prices = future_prices.values.reshape(-1)
            trend = self._get_trend(future_prices, price)
            obs_point = [price, profit, trend]

            # 2. Сохранить ее в общий декью
            self.history.append(obs_point)
            self.last_update = ts

    @staticmethod
    def _get_trend(future_prices, current_price):
        diff = np.array(future_prices / current_price - 1) * 100
        weights = np.linspace(1.0, 0.5, len(diff))
        trend_indicator = np.average(diff, weights=weights)
        return trend_indicator

    def _get_profit(self):
        if self.context.get("is_open", domain="Trade"):
            current_price = self.context.get("highest_bid")
            profit = current_price / self.context.get("open_price", domain="Trade") - 1 - self.context.market_fee
            profit = profit * 10
        else:
            profit = 0
        return profit

