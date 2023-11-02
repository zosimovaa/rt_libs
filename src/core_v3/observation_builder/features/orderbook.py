"""
Orderbook features_del

"""
import logging
import numpy as np

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class OrderbookDiffFeature(BaseFeature):
    def __init__(self, *args, **kwargs):
        self.level = kwargs.pop("level", 0.001)
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")

        asks = data_point.get_values("asks_" + str(self.level), period=self.period, agg="sum")
        bids = data_point.get_values("bids_" + str(self.level), period=self.period, agg="sum")

        feature = bids - asks

        feature = feature / np.abs(feature).mean()
        return feature
