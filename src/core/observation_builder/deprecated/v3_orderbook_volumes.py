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

from core.observation_builder.interface import ObservationBuilderInterface
from core.observation_builder.features import TradeStateFeature
from core.observation_builder.features import OrderbookDiffFeature2D
from core.observation_builder.features import ProfitFeature
from core.observation_builder.features import TradeBalanceFeature
from core.observation_builder.features import Rates2DFactorFeature


logger = logging.getLogger(__name__)


class ObservationBuilderOrderbookTradeBalance(ObservationBuilderInterface):
    def __init__(self, context, step_factor=(1, 3, 12), levels=None):
        """Конструктор класса"""
        self.context = context
        self.trade_state_feat = TradeStateFeature(context)
        self.rate_feat = Rates2DFactorFeature(context, step_factor=step_factor)
        self.profit_feat = ProfitFeature(context)
        self.orderbook_feat = OrderbookDiffFeature2D(context, levels=levels)
        self.balance_feat = TradeBalanceFeature(context)

    def reset(self):
        """Сброс параметров"""
        self.trade_state_feat.reset()
        self.rate_feat.reset()
        self.profit_feat.reset()
        self.orderbook_feat.reset()
        self.balance_feat.reset()

    def get(self, data_point):
        trade_state = self.trade_state_feat.get()
        rates2d = self.rate_feat.get()
        profit = self.profit_feat.get()
        orderbook2d = self.orderbook_feat.get()
        trade_balance = self.balance_feat.get()

        # ------------------------------------------
        # observation
        static_data = [trade_state]

        conv_data = np.concatenate([
            rates2d,
            profit.reshape(-1, 1),
            orderbook2d,
            trade_balance.reshape(-1, 1)
        ], axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(conv_data, dtype=np.float32)
        ]
        return observation
