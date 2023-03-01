"""
Orderbook features_del

"""
import logging
import numpy as np

from .abstract_feature import AbstractFeature

logger = logging.getLogger(__name__)


class OrderbookDiffFeature2D(AbstractFeature):
    def __init__(self, context, step_factor=(1,), scale_output=0.3, levels=(0.001, 0.0025, 0.005, 0.0075)):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)
        self.levels = levels

    def _get(self):
        data_point = self.context.data_point
        obs = []

        for sf in self.step_factor:

            for level in self.levels:
                asks = data_point.get_values("asks_" + str(level), step_factor=sf)
                bids = data_point.get_values("bids_" + str(level), step_factor=sf)

                feature = bids - asks
                if sf > 1:
                    res = []
                    for i in range(data_point.observation_len):
                        val = feature[sf * i: sf * (i + 1)]
                        res.append(val.sum())
                    feature = np.array(res)

                feature = feature / np.abs(feature).mean() * self.scale_output
                feature = feature.reshape(-1, 1)

                obs.append(feature)

        feature = np.concatenate([*obs], axis=1)

        return feature
