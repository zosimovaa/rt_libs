"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class RatesFeature(BaseFeature):
    """Возвращает значения курса, нормализованные к текущему"""

    def __init__(self, *args, **kwargs):
        self.price = kwargs.pop("price", "highest_bid")
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")
        current_price = self.context.get(self.price)
        data = data_point.get_values(self.price, period=self.period)
        feature = data / current_price - 1
        return feature


class RatesFeatureNorm(BaseFeature):
    """Возвращает значения курса, нормализованные к среднему за период"""

    def __init__(self, *args, **kwargs):
        self.price = kwargs.pop("price", "highest_bid")
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")
        data = data_point.get_values(self.price, period=self.period)
        norm_values = data_point.get_values(self.price, period=self.period, num=-1)
        feature = data / np.mean(norm_values) - 1
        return feature


class RatesDiffFeature(BaseFeature):

    def __init__(self, *args, **kwargs):
        self.price = kwargs.pop("price", "highest_bid")
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")
        current_price = self.context.get(self.price)

        data = data_point.get_values(self.price, period=self.period, num=data_point.observation_len + 1)
        data = data.diff()
        feature = data / current_price

        return feature
