"""

"""

import logging
import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class ProfitStateSingleFeature(BaseFeature):
    """Returns a matrix with normalized rates for different periods"""

    def __init__(self, *args, **kwargs):
        self.edge = kwargs.pop("edge", 0)  # значение, при +/- котором считаем, что профит выше или ниже нормы
        super().__init__(*args, **kwargs)

    def _get(self):
        profit = self.context.get("profit")
        if profit >= self.edge:
            feature = 1
        elif profit <= -self.edge:
            feature = -1
        else:
            feature = 0
        return np.array([feature])
