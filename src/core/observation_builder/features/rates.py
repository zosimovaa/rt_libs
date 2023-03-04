"""
Module with basic features_del - deal status, rate, profit.
"""

import logging
import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class RatesFeature2D(BaseFeature):
    """Returns a matrix with normalized rates for different periods"""

    def __init__(self, context, step_factor=1, scale_output=30):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        data_point = self.context.data_point
        current_price = self.context.get("highest_bid")

        feature = data_point.get_values("highest_bid", step_factor=self.step_factor)
        feature = (feature / current_price - 1) * self.scale_output

        if self.step_factor > 1:
            res = []
            for i in range(data_point.observation_len):
                val = feature[self.step_factor * i: self.step_factor * (i + 1)]
                res.append(val.mean())
            feature = np.array(res).reshape(-1, 1)
        else:
            feature = feature.reshape(-1, 1)
        return feature
