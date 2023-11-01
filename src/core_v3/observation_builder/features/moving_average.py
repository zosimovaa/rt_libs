"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)

def moving_average(a, n=2):
    ret = np.cumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class MovingAverageFeature(BaseFeature):
    """Rates feature normalized with current rate"""

    def __init__(self, *args, **kwargs):
        self.ma_points = kwargs.pop("ma_points", 1)
        self.feature = kwargs.pop("feature", "highest_bid")

        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")
        num = data_point.observation_len + self.ma_points - 1
        current_value = data_point.get_value(self.feature)

        values = data_point.get_values(self.feature, step_factor=self.period, num=num)
        feature = moving_average(values, n=self.ma_points)

        # нормализуем.... надо протестить как это будет выглядеть
        feature = feature / current_value - 1

        return feature


