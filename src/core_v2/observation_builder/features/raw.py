"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class RawValueFeature(BaseFeature):
    """Извлечение значения как есть из датапоинта"""
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "feature")
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")
        feature = data_point.get_value(self.name, step_factor=self.step_factor)
        return np.array([feature])


class RawContextFeature(BaseFeature):
    """Извлечение значения как есть из контекста"""
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "feature")
        super().__init__(*args, **kwargs)

    def _get(self):
        feature = self.context.get(self.name)
        return np.array([feature])