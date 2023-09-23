"""
Orderbook features_del

"""
import logging
import numpy as np

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class TradeVolumes2D(BaseFeature):
    def __init__(self, context, step_factor=1, scale_output=0.2):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        data_point = self.context.data_point

        sell_vol = data_point.get_values("sell_vol", step_factor=self.step_factor)
        buy_vol = data_point.get_values("buy_vol", step_factor=self.step_factor)
        feature = buy_vol - sell_vol

        if self.step_factor > 1:
            res = []
            for i in range(data_point.OBSERVATION_LEN):
                val = feature[self.step_factor * i: self.step_factor * (i + 1)]
                res.append(val.sum())
            feature = np.array(res)

        feature = feature / np.abs(feature).mean() * self.scale_output

        return feature.reshape(-1, 1)


class TradeCount2D(BaseFeature):
    def __init__(self, context, step_factor=1, scale_output=0.2):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        data_point = self.context.data_point

        sell_vol = data_point.get_values("sell_num", step_factor=self.step_factor)
        buy_vol = data_point.get_values("buy_num", step_factor=self.step_factor)
        feature = buy_vol-sell_vol

        if self.step_factor > 1:
            res = []
            for i in range(data_point.OBSERVATION_LEN):
                val = feature[self.step_factor * i: self.step_factor * (i + 1)]
                res.append(val.sum())
            feature = np.array(res)
        feature = feature / np.abs(feature).mean() * self.scale_output

        return feature.reshape(-1, 1)
