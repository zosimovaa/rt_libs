"""
Orderbook features_del

"""
import logging
import numpy as np

from .abstract_feature import AbstractFeature

logger = logging.getLogger(__name__)


class TradeVolumes2D(AbstractFeature):
    def __init__(self, context, step_factor=(1,), scale_output=0.2):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        data_point = self.context.data_point
        obs = []

        for sf in self.step_factor:

            sell_vol = data_point.get_values("sell_vol", step_factor=sf)
            buy_vol = data_point.get_values("buy_vol", step_factor=sf)
            feature = buy_vol - sell_vol

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


class TradeCount2D(AbstractFeature):
    def __init__(self, context, step_factor=(1,), scale_output=0.2):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)

    def _get(self):
        data_point = self.context.data_point
        obs = []

        for sf in self.step_factor:

            sell_vol = data_point.get_values("sell_num", step_factor=sf)
            buy_vol = data_point.get_values("buy_num", step_factor=sf)
            feature = buy_vol-sell_vol

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
