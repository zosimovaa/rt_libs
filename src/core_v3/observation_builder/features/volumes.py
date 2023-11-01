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

        sell_vol_norm = data_point.get_values("sell_vol", step_factor=self.period, num=-1, agg="sum")
        buy_vol_norm = data_point.get_values("buy_vol", step_factor=self.period, num=-1, agg="sum")
        norm_values = np.concatenate([sell_vol_norm, buy_vol_norm])
        norm_value = np.array(norm_values, dtype=np.float32).mean()

        sell_vol = data_point.get_values("sell_vol", step_factor=self.period, agg="sum")
        buy_vol = data_point.get_values("buy_vol", step_factor=self.period, agg="sum")
        feature = buy_vol - sell_vol

        feature = feature / norm_value
        return feature


class TradeCount(BaseFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get(self):
        data_point = self.context.get("data_point")

        sell_vol_norm = data_point.get_values("sell_num", step_factor=self.period, num=-1, agg="sum")
        buy_vol_norm = data_point.get_values("buy_num", step_factor=self.period, num=-1, agg="sum")
        norm_values = np.concatenate([sell_vol_norm, buy_vol_norm])
        norm_value = np.array(norm_values, dtype=np.float32).mean()

        sell_vol = data_point.get_values("sell_num", step_factor=self.period, agg="sum")
        buy_vol = data_point.get_values("buy_num", step_factor=self.period, agg="sum")
        feature = buy_vol - sell_vol

        feature = feature / norm_value
        return feature
