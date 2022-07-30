"""
 Билдер с тремя фичами:
 - состояние сделки
 - данные для НС
   - курс
   - профит
   -

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


class ObservationBuilderBasic(ObservationBuilderInterface):
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