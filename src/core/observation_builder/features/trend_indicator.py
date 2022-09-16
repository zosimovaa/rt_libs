"""
Модуль с реализацией фичи trend indicator.
"""
import logging
import numpy as np

from .abstract_class import AbstractFeatureWithHistory

logger = logging.getLogger(__name__)


class TrendIndicatorFeature(AbstractFeatureWithHistory):

    def __init__(self, context, ti_decrease_coef_start=1., ti_decrease_coef_end=.5):
        super().__init__(context)
        self.ti_decrease_coef_start = ti_decrease_coef_start
        self.ti_decrease_coef_end = ti_decrease_coef_end

    def _build_point(self, dp, cursor=None):
        if cursor is None:
            cursor = dp.get_indexes(num=1)

        if dp.fut_len > 0:

            current_value = dp.get_value("highest_bid", cursor=cursor)
            future_values = dp.get_values(name="highest_bid", cursor=cursor, num=-dp.fut_len)

            diff = np.array(future_values / current_value - 1) * 100
            coeffs = np.linspace(self.ti_decrease_coef_start, self.ti_decrease_coef_end, len(diff))

            trend_indicator = np.average(diff, weights=coeffs)
            return trend_indicator
        else:
            return 0
