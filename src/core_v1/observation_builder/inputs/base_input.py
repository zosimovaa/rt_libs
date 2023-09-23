"""
fdd
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseInput(ABC):
    """
    Базовый класс инпута
    """
    def __init__(self, *features):
        self.features = features

    def reset(self):
        """Сброс инпута в начаольное состояние"""
        for feat in self.features:
            feat.reset()

    @abstractmethod
    def get(self):
        """Метод возвращает сформированынй инпут"""
        raise NotImplementedError
