import logging

logger = logging.getLogger(__name__)


class BaseInput:
    def __init__(self, **features):
        self.features = features

    def reset(self):
        """Сброс параметров"""
        for feat in self.features:
            feat.reset()

    def get(self):
        raise NotImplementedError
