import logging
import numpy as np
from collections import deque

from .interface import ObservationBuilderInterface

logger = logging.getLogger(__name__)


class ObservationBuilderBasic_deprecated(ObservationBuilderInterface):
    """Билдер с 2-мя фичами"""

    SCALE_FACTOR = 10

    def __init__(self, context):
        self.context = context

    def reset(self):
        pass

    def get(self, data_point):
        # trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # rates representation
        current_price = self.context.get("highest_bid")
        rates = (data_point.get_values("highest_bid").values / current_price - 1) * self.SCALE_FACTOR

        # profit representation
        profit = self._get_profit(data_point, self.context.trade)

        # observation
        conv_data = np.concatenate([rates.reshape(-1, 1), profit.reshape(-1, 1)], axis=1)
        observation = [
            np.array([trade_state], dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

    def _get_profit(self, data_point, trade):
        timestamps = data_point.get_timestamps()

        if trade is not None:
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
            current_rates = data_point.get_values("highest_bid").values
            profit = current_rates / trade.open_price - 1 - self.context.market_fee
            profit = profit * mask * self.SCALE_FACTOR
        else:
            profit = np.zeros(len(timestamps))
        return profit





class ObservationBuilderTrendIndicator_old(ObservationBuilderInterface):
    """Билдер с 2-мя фичами"""

    SCALE_FACTOR = 10
    TI_DECREASE_COEF_START = 1.
    TI_DECREASE_COEF_END = .5

    def __init__(self, context):
        """Конструктор класса"""
        self.context = context
        self.ti = None
        self.ti_date = 0

    def reset(self):
        """Сброс параметров"""
        self.ti = None
        self.ti_date = 0

    def get(self, data_point):
        # trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # rates representation
        current_price = self.context.get("highest_bid")
        rates = (data_point.get_values("highest_bid").values / current_price - 1) * self.SCALE_FACTOR

        # profit representation
        profit = self._get_profit(data_point, self.context.trade)

        # trend indicator representation
        trend = self._get_trend_indicator(data_point)

        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            trend.reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

    def _get_profit(self, data_point, trade):
        timestamps = data_point.get_timestamps()

        if trade is not None:
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
            current_rates = data_point.get_values("highest_bid").values
            profit = current_rates / trade.open_price - 1 - self.context.market_fee
            profit = profit * mask * self.SCALE_FACTOR
        else:
            profit = np.zeros(len(timestamps))
        return profit

    def _get_trend_indicator(self, dp):

        if self.ti is None:
            # Запрос первого наблюдения
            self.ti = self._build_ti(dp)
            self.ti_date = dp.get_current_ts()

        elif dp.get_current_ts() > self.ti_date + dp.period:
            # Были потери в данных
            self.ti = self._build_ti(dp)
            self.ti_date = dp.get_current_ts()

        elif dp.get_current_ts() == self.ti_date:
            # Повторное построение observation
            pass

        else:
            # Запрос следующей точки
            ti = self._get_ti(dp)
            self.ti.append(ti)
            self.ti_date = dp.get_current_ts()

        return np.array(self.ti)

    def _get_ti(self, dp, cursor=None):
        if cursor is None:
            cursor = dp.get_current_ts()

        if dp.fut_len > 0:

            current_value = dp.get_value("highest_bid", cursor=cursor)
            future_values = dp.get_future_values("highest_bid", cursor=cursor).values

            diff = np.array(future_values / current_value - 1) * 100
            coeffs = np.linspace(self.TI_DECREASE_COEF_START, self.TI_DECREASE_COEF_END, len(diff))

            trend_indicator = np.average(diff, weights=coeffs)
            return trend_indicator
        else:
            return 0

    def _build_ti(self, dp):
        ti_arr = deque(maxlen=len(dp.get_timestamps()))
        indexes = dp.get_timestamps()
        for cursor in indexes:
            ti = self._get_ti(dp, cursor=cursor)
            ti_arr.append(ti)
        return ti_arr

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
        rates = (data_point.get_values("highest_bid").values / current_price - 1) * 10

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
            profit = data_point.get_values("highest_bid").values.copy().reshape(-1)
            mask = data_point.get_timestamps() > self.context.get("open_ts", domain="Trade")
            profit = profit / self.context.get("open_price", domain="Trade") - 1 - self.context.market_fee
            profit = profit * mask * 10
        else:
            profit = np.zeros(data_point.offset)
        return profit

    def _get_trend(self):
        ti = []
        # todo - потери времени на цикле.
        #for i in range(self.context.data_point.offset):
        for i in self.context.data_point.data.index:
            current_value = self.context.data_point.get_value("highest_bid", cursor=i)

            # todo Здесь похоже, что есть ошибка
            future_values = self.context.data_point.get_future_values("highest_bid").values.reshape(-1)

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
            price = self.context.data_point.get_value("highest_bid", cursor=idx)
            future_values = self.context.data_point.get_future_values("highest_bid").values.reshape(-1)
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

            future_prices = self.context.data_point.get_future_values("highest_bid")
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


class ObservationBuilderV2TradesSimpleBalance(ObservationBuilderInterface):
    """
    Билдер с данными об объемах.
    В качестве trade feature - простой баланс. Расчет идет в БД - (buy-sell)/(buy+sell)
    """
    SCALE_FACTOR = 10
    TRADE_VOLUMES_SCALE_FACTOR = 0.1
    TI_DECREASE_COEF_START = 1.
    TI_DECREASE_COEF_END = .5

    def __init__(self, context):
        """Конструктор класса"""
        self.context = context

    def reset(self):
        """Сброс параметров"""
        pass

    def get(self, data_point):
        # Trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # Rate
        current_price = self.context.get("highest_bid")
        rates = (data_point.get_values("highest_bid").values / current_price - 1) * self.SCALE_FACTOR

        # Profit
        profit = self._get_profit(data_point, self.context.trade)

        # Trade feature
        trades = self._get_trade_feature(data_point)

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            trades.reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

    def _get_profit(self, data_point, trade):
        timestamps = data_point.get_timestamps()

        if trade is not None:
            mask = (timestamps > trade.open_ts) & (timestamps <= trade.close_ts)
            current_rates = data_point.get_values("highest_bid").values
            profit = current_rates / trade.open_price - 1 - self.context.market_fee
            profit = profit * mask * self.SCALE_FACTOR
        else:
            profit = np.zeros(len(timestamps))
        return profit

    def _get_trade_feature(self, data_point):
        return data_point.get_values("balance").values


class ObservationBuilderV2TradesBuySellFeats(ObservationBuilderV2TradesSimpleBalance):
    """
    Билдер с данными об объемах.
    В качестве trade feature - простой баланс. Расчет идет в БД - (buy-sell)/(buy+sell)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, data_point):
        # Trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # Rate
        current_price = self.context.get("highest_bid")
        rates = (data_point.get_values("highest_bid").values / current_price - 1) * self.SCALE_FACTOR

        # Profit
        profit = self._get_profit(data_point, self.context.trade)

        # Trade feature
        buy_vol_rel, sell_vol_rel = self._get_trade_feature(data_point)

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1),
            buy_vol_rel.reshape(-1, 1),
            sell_vol_rel.reshape(-1, 1),
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation

    def _get_trade_feature(self, data_point):
        total_avg_volume = data_point.get_hist_values("total_vol").mean()

        if total_avg_volume:
            buy_vol_rel = data_point.get_values("buy_vol").values / total_avg_volume * self.TRADE_VOLUMES_SCALE_FACTOR
            sell_vol_rel = data_point.get_values("sell_vol").values / total_avg_volume * self.TRADE_VOLUMES_SCALE_FACTOR
        else:
            buy_vol_rel = np.zeros(len(data_point.get_timestamps()))
            sell_vol_rel = np.zeros(len(data_point.get_timestamps()))
        return buy_vol_rel, sell_vol_rel


class ObservationBuilderV2TradesRelativeBalance(ObservationBuilderV2TradesSimpleBalance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_trade_feature(self, data_point):
        total_avg_volume = data_point.get_hist_values("total_vol").mean()

        if total_avg_volume:
            buy_vols = data_point.get_values("buy_vol").values
            sell_vols = data_point.get_values("sell_vol").values

            balance = buy_vols - sell_vols

            balance_rel = balance // total_avg_volume

        else:
            balance_rel = np.zeros(len(data_point.get_timestamps()))
        return balance_rel




