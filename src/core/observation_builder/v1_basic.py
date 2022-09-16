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

from .interface import ObservationBuilderInterface
from .features import TradeStateFeature
from .features import Rates1DFeature
from .features import Rates2DFactorFeature
from .features import ProfitFeature


logger = logging.getLogger(__name__)


class ObservationBuilderBasic(ObservationBuilderInterface):
    def __init__(self, context):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates1DFeature(context)
        self.profit_feat = ProfitFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates = self.rate_feat.get()
        profit = self.profit_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates.reshape(-1, 1),
            profit.reshape(-1, 1)
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation


class ObservationBuilderBasicMultiRate(ObservationBuilderInterface):
    def __init__(self, context, step_factor=(1, 3, 12)):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates2DFactorFeature(context, step_factor=step_factor)
        self.profit_feat = ProfitFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates2d = self.rate_feat.get()
        profit = self.profit_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates2d,
            profit.reshape(-1, 1)
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation


class ObservationBuilderBasicMultiRateSeparate(ObservationBuilderInterface):
    def __init__(self, context, step_factor=(1, 3, 12)):
        """Конструктор класса"""
        if step_factor is None:
            step_factor = [1, 3, 12]
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates2DFactorFeature(context, step_factor=step_factor)
        self.profit_feat = ProfitFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates2d = self.rate_feat.get()
        profit = self.profit_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]
        conv_rates = rates2d
        conv_profit = profit.reshape(-1, 1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_rates, dtype=np.float32),
            np.array(conv_profit, dtype=np.float32)
        ]
        return observation