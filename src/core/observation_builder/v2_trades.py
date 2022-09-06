"""
Билдер с тремя фичами:
 - состояние сделки
 - данные для НС
   - курс
   - профит
   - баланс

Данные для сети подаются в формате [num_of_points, num_features]
        rate perp        profit repr
array([[-1.9651896e-01,  0.0000000e+00],
       [-1.8928000e-01,  0.0000000e+00],
       [-1.8809754e-01,  0.0000000e+00]]
"""

import logging
import numpy as np
from .interface import ObservationBuilderInterface

logger = logging.getLogger(__name__)


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



