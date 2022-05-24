import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class ObservationBuilderInterface(ABC):
    """Описывает внешний контракт взаимодействия. Интерфейс."""
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class ObservationBuilder(ObservationBuilderInterface):
    """Абстрактный билдер с базовыми полями. Заготовка для реализации."""
    def __init__(self, context):
        ObservationBuilderInterface.__init__(self)
        self.context = context
        self.history = None
        self.last_update = 0

    def get(self):
        pass

    def reset(self):
        self.history = None
        self.last_update = 0


class ObservationBuilderBasic(ObservationBuilder):
    """Билдер с 2-мя фичами. Без кэша."""
    def __init__(self, context):
        ObservationBuilder.__init__(self, context)

    def get(self):
        data_point = self.context.data_point
        profit = self._get_profit()

        current_price = self.context.get("highest_bid")

        rates = (data_point.get_prices("highest_bid").values / current_price - 1) * 10

        conv_data = np.concatenate([rates.reshape(-1, 1), profit.reshape(-1, 1)], axis=1)
        observation = [
            np.array([self.context.get("is_open", domain="Trade")], dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

    def _get_profit(self):
        data_point = self.context.data_point
        if self.context.get("is_open", domain="Trade"):
            profit = data_point.get_prices("highest_bid").values.copy().reshape(-1)
            mask = data_point.get_timestamps() > self.context.get("open_ts", domain="Trade")
            profit = profit / self.context.get("open_price", domain="Trade") - 1 - self.context.market_fee
            profit = profit * mask * 10
        else:
            profit = np.zeros(data_point.offset)
        return profit


class ObservationBuilderBasicCache(ObservationBuilder):
    """Билдер с 2-мя фичами. С кэшом. Требует инициализации (сброс с датапоинтом в контексте)."""
    def __init__(self, context):
        ObservationBuilder.__init__(self, context)

    def init_history(self):
        buffer_len = self.context.data_point.offset
        self.history = deque(maxlen=buffer_len)

        if buffer_len > 1:
            for i in range(buffer_len):
                price = self.context.data_point.get_price("highest_bid", cursor=i)
                profit = 0
                obs_point = [price, profit]
                self.history.append(obs_point)

    def get(self):
        if self.history is None:
            self.init_history()

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


class ObservationBuilderFutureFeature(ObservationBuilderBasic):
    """Билдер с добавлением признака тренда. Без кэша."""
    def __init__(self, context):
        ObservationBuilderBasic.__init__(self, context)

    def get(self):
        profit = self._get_profit()
        trend = self._get_trend()

        current_price = self.context.get("highest_bid")
        rates = (self.context.data_point.get_prices("highest_bid").values / current_price - 1) * 10

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            trend.reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array([self.context.get("is_open", domain="Trade")], dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

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


class ObservationBuilderFutureFeatureCache(ObservationBuilderBasicCache):
    """Билдер с добавлением признака тренда. С кэшом. Требует инициализации (сброс с датапоинтом в контексте)."""
    def __init__(self, context):
        ObservationBuilderBasicCache.__init__(self, context)

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

    def get(self):
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
