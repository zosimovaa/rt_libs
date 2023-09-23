"""
Orderbook features_del

"""
import logging
import numpy as np

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class TradeVolumes(BaseFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")

        sell_vol = data_point.get_values("sell_vol", step_factor=self.step_factor)
        buy_vol = data_point.get_values("buy_vol", step_factor=self.step_factor)
        feature = buy_vol - sell_vol

        if self.step_factor > 1:
            feature = np.sum(feature.reshape(-1, self.step_factor), axis=1)

        feature = feature / np.abs(feature).mean()
        return feature


class TradeCount(BaseFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")

        sell_vol = data_point.get_values("sell_num", step_factor=self.step_factor)
        buy_vol = data_point.get_values("buy_num", step_factor=self.step_factor)
        feature = buy_vol-sell_vol

        if self.step_factor > 1:
            feature = np.sum(feature.reshape(-1, self.step_factor), axis=1)

        feature = feature / np.abs(feature).mean()

        return feature
