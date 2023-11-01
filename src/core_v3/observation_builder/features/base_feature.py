import logging
import numpy as np

from ...context import ContextConsumer

logger = logging.getLogger(__name__)


class BaseFeature(ContextConsumer):
    """Базовый класс, рассчитывает значения на лету"""
    ATTRIBUTES_NAMES = ("step_factor", "level", "name", "feature", "ma_points")

    def __init__(self, alias, period=300, scale_output=1, clip_output=0, normalization=False):
        super().__init__(alias)
        self.period = period
        self.scale_output = scale_output
        self.clip_output = clip_output
        self.normalization = normalization

    def get(self) -> np.ndarray:
        data = self._get()

        # data normalization or scaling
        if self.normalization:
            total = np.abs(data).sum()
            if total > 0:
                data = data / np.abs(data).mean()

        # data scaling
        if self.scale_output == 1:
            pass
        else:
            data = data * self.scale_output

        # data clip
        if self.clip_output:
            data = np.clip(data, -self.clip_output, self.clip_output)
        return data

    def reset(self) -> None:
        """Reset to initial state. Relevant for the heirs of AbstractFeatureWithHistory"""
        pass

    def _get(self) -> np.ndarray:
        """The method returns the current feature value for the given datapoint"""
        raise NotImplementedError

    def __str__(self):
        obj = self.__dict__
        feature_name = self.__class__.__name__
        for attr_name in self.ATTRIBUTES_NAMES:
            if attr_name in obj:
                feature_name += f"-{attr_name}:{obj[attr_name]}"
        feature_name

        return feature_name
