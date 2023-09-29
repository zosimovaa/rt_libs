"""
Orderbook features_del

"""
import logging
import numpy as np

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class OrderbookDiffFeature2D(BaseFeature):
    def __init__(self, context, step_factor=1, scale_output=0.3, level=0.001):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)
        self.level = level

    def _get(self):
        data_point = self.context.data_point

        asks = data_point.get_values("asks_" + str(self.level), step_factor=self.step_factor)
        bids = data_point.get_values("bids_" + str(self.level), step_factor=self.step_factor)

        feature = bids - asks
        if self.step_factor > 1:
            res = []
            for i in range(data_point.observation_len):
                val = feature[self.step_factor * i: self.step_factor * (i + 1)]
                res.append(val.sum())
            feature = np.array(res)

        feature = feature.reshape(-1, 1)
        feature = feature / np.abs(feature).mean() * self.scale_output

        return feature.reshape(-1, 1)
