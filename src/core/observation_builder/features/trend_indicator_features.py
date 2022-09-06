from .i_abstract_feature import AbstractFeatureWithHistory
import logging
import numpy as np


logger = logging.getLogger(__name__)


class TrendIndicatorFeature(AbstractFeatureWithHistory):
    TI_DECREASE_COEF_START = 1.
    TI_DECREASE_COEF_END = .5

    def __init__(self, context):
        super().__init__(context)

    def _build_point(self, dp, cursor=None):
        if cursor is None:
            cursor = dp.get_current_ts()

        if dp.fut_len > 0:

            current_value = dp.get_value("highest_bid", cursor=cursor)
            future_values = dp.get_future_values("highest_bid", cursor=cursor).values

            diff = np.array(future_values / current_value - 1) * 100
            coeffs = np.linspace(self.TI_DECREASE_COEF_START, self.TI_DECREASE_COEF_END, len(diff))

            trend_indicator = np.average(diff, weights=coeffs)
            return trend_indicator
        else:
            return 0
