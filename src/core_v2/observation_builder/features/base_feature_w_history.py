import logging
import numpy as np
from collections import deque

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class BaseFeatureWithHistory(BaseFeature):
    """
    For hard-to-calculate features. For each step, only the actual data point is calculated. The previous ones are stored in the cache.
    But it is better to use precompute features for precomputation in the dataset, if possible.
    """

    def __init__(self, alias="train", step_factor=(1,), scale_output=1):
        super().__init__(alias, step_factor=step_factor, scale_output=scale_output)
        self.data = None
        self.last_update = None

    def _build_point(self, dp, cursor=None):
        """The method must implement the calculation of the data point for the given cursor. If cursor is not defined, then """
        pass

    def _get(self):
        """Defines the data generation logic"""
        dp = self.context.get("data_point")

        # First Observation Request
        if self.data is None:
            self.data = self._init_data(dp)
            self.last_update = dp.get_index()

        # There were data losses
        elif dp.get_index() > self.last_update + dp.period:
            self.data = self._init_data(dp)
            self.last_update = dp.get_index()

        # Rebuild observation
        elif dp.get_index() == self.last_update:
            pass

        # Request next point
        else:
            point = self._build_point(dp)
            self.data.append(point)
            self.last_update = dp.get_index()

        data = np.array(self.data)

        return data

    def reset(self):
        super().reset()
        self.data = None
        self.last_update = None

    def _init_data(self, dp):
        """При переходе в вычислениям со step_factor встал вопрос как рассчитывать точки наблюдения с историей.
        Оно пока не работает
        """
        raise NotImplementedError

        #indexes = dp.get_indexes()
        #data = deque(maxlen=len(indexes))
        #for cursor in indexes:
        #    point = self._build_point(dp, cursor=cursor)
        #    data.append(point)
        #return data

