"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class RawValueFeature1D(BaseFeature):
    """Profit calculation"""
    def __init__(self, context, step_factor=1, scale_output=1, name="feature"):
        self.name = name
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        data_point = self.context.data_point
        feature = data_point.get_value(self.name, step_factor=self.step_factor)[0]
        return np.array([feature])


class RawContextFeature1D(BaseFeature):
    """Profit calculation"""
    def __init__(self, context, step_factor=1, scale_output=1, name="feature"):
        self.name = name
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        feature = self.context.get(self.name)
        return np.array([feature])