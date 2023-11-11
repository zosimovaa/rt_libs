import abc
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class BaseFeature(metaclass=abc.ABCMeta):
    """Calculate features on the fly"""

    def __init__(self, context, step_factor=(1,), scale_output=1):
        self.context = context
        self.scale_is_broken = False
        self.step_factor = step_factor
        self.scale_output = scale_output

    def get(self, norm=False, clip_edge=4) -> np.ndarray:
        data = self._get()

        # data normalization
        if norm:
            data = data / np.abs(data).mean()

        # Check data scale
        if clip_edge:
            check_data = np.abs(data) > clip_edge
            if check_data.sum():
                self.scale_is_broken = True
                data = np.clip(data, -clip_edge, clip_edge)
        return data

    def reset(self) -> None:
        """Reset to initial state. Relevant for the heirs of AbstractFeatureWithHistory"""
        self.scale_is_broken = False

    @abc.abstractmethod
    def _get(self) -> np.ndarray:
        """The method returns the current feature value for the given datapoint"""
        raise NotImplementedError


class BaseFeatureWithHistory(BaseFeature):
    """
    For hard-to-calculate features. For each step, only the actual data point is calculated. The previous ones are stored in the cache.
    But it is better to use precompute features for precomputation in the dataset, if possible.
    """

    def __init__(self, context, step_factor=(1,), scale_output=1):
        super().__init__(context, step_factor=step_factor, scale_output=scale_output)
        self.data = None
        self.last_update = None

    @abc.abstractmethod
    def _build_point(self, dp, cursor=None):
        """The method must implement the calculation of the data point for the given cursor. If cursor is not defined, then """
        pass

    def _get(self):
        """Defines the data generation logic"""
        dp = self.context.data_point

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
