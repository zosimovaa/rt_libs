"""
МОдуль
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseInput:
    """Базовый класс инпута"""
    def __init__(self, *features):
        self.features = features

    def reset(self):
        """Сброс инпута в начальное состояние"""
        for feat in self.features:
            feat.reset()

    def get(self):
        """Метод возвращает сформированынй инпут"""
        raise NotImplementedError


class Input1D(BaseInput):
    """Одномерный инпут"""
    def __init__(self, *features):
        super().__init__(*features)

    def get(self):
        data = np.concatenate([feat.get() for feat in self.features], axis=0)
        return np.array(data, dtype=np.float32)


class Input2D(BaseInput):
    """Двухмерный инпут"""
    def __init__(self, *features):
        super().__init__(*features)

    def get(self):
        data = np.concatenate([feat.get().reshape(-1, 1) for feat in self.features], axis=1)
        return np.array(data, dtype=np.float32)
