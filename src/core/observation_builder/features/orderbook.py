"""
Orderbook features_del

"""
import logging
import numpy as np

from .abstract_class import AbstractFeature

logger = logging.getLogger(__name__)


class OrderbookDiffFeature2D(AbstractFeature):
    def __init__(self, context, levels=(0.001, 0.0025, 0.005, 0.0075)):
        super().__init__(context)
        self.levels = levels

    def _get(self):
        data_point = self.context.data_point
        result = []

        for level in self.levels:
            asks = data_point.get_values("asks_" + str(level))
            bids = data_point.get_values("bids_" + str(level))

            feature = bids - asks
            feature = feature / np.abs(feature).mean()

            result.append(feature.reshape(-1, 1))
        obs = np.concatenate(result, axis=1)
        return obs
