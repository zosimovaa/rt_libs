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


class ObservationBuilderBasic(ObservationBuilderInterface):
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

        # observation
        conv_data = np.concatenate([rates.reshape(-1, 1), profit.reshape(-1, 1)], axis=1)
        observation = [
            np.array([trade_state], dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]

        return observation

    def _get_profit(self, data_point, trade_state):
        """Считает профит на лету, по данным текущего datapoint"""
        if trade_state:
            profit = data_point.get_prices("highest_bid").values.copy().reshape(-1)
            mask = data_point.get_timestamps() > self.context.get('open_ts', domain="Trade")
            profit = profit / self.context.get("open_price", domain="Trade") - 1 - self.context.market_fee
            profit = profit * mask * 10
        else:
            profit = np.zeros(data_point.offset)
        return profit


class ObservationBuilderBasicCache(ObservationBuilderInterface):
    """Билдер с 2-мя фичами. С кэшом. Работает быстрее, чем без кэша. Критично для обучения.
    При добавлении новой точки в историю:
        - нет проверки на валидность последовательности, что ничего не пропушено.
        При обучении не критично, но при реальной торговле такое недопустимо.
        - есть защита от повторного формирования точки в кэше.
    """
    def __init__(self, context):

        self.context = context
        self.history = None     # Кэш истории
        self.last_update = 0    # Контроль апдейта кэша

    def reset(self):
        self.history = None
        self.last_update = 0

    def init_history(self, data_point):
        """Начальная инициализация истории в кэше"""
        buffer_len = data_point.offset
        self.history = deque(maxlen=buffer_len)

        if buffer_len > 1:
            for i in range(buffer_len):
                price = data_point.get_price("highest_bid", cursor=i)
                profit = 0
                obs_point = [price, profit]
                self.history.append(obs_point)

    def get(self, data_point):
        if self.history is None:
            self.init_history(data_point)

        self._build_history_point()

        history = np.array(self.history)

        # price feature
        rates = history[:, 0]
        current_price = self.context.get("highest_bid")
        rates = (rates / current_price - 1) * 10

        # profit feature
        profits = history[:, 1]

        # conv_date
        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profits.reshape(-1, 1)
        ], axis=1)

        observation = [
            np.array([self.context.get("is_open", domain="Trade")], dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]

        return observation

    def _build_history_point(self):
        ts = self.context.get("ts", default=-1)
        if self.last_update != ts:
            price = self.context.get("highest_bid")
            profit = self._get_profit()

            obs_point = [price, profit]

            self.history.append(obs_point)
            self.last_update = ts

    def _get_profit(self):
        if self.context.get("is_open", domain="Trade"):
            current_price = self.context.get("highest_bid")
            profit = current_price / self.context.get("open_price", domain="Trade") - 1 - self.context.market_fee
            profit = profit * 10
        else:
            profit = 0
        return profit
