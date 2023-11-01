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
        name = kwargs.pop("name", "feature")
        super().__init__(*args, **kwargs)
        self.name = name

    def _get(self):
        data_point = self.context.get("data_point")
        feature = data_point.get_value(self.name, step_factor=self.period)
        return np.array([feature], dtype=np.float32)

    def __str__(self):
        return f"RawValue_{self.name}(sf:{self.period})"


class RawContextFeature(BaseFeature):
    """Извлечение значения как есть из контекста"""
    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "feature")
        super().__init__(*args, **kwargs)
        self.name = name

    def _get(self):
        feature = self.context.get(self.name)
        return np.array([feature], dtype=np.float32)

    def __str__(self):
        return f"RawContext_{self.name}(sf:{self.period})"
