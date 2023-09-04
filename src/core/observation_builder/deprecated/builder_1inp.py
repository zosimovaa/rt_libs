"""
Базовый билдер с фичей в одном измерении:

"""

import logging
import numpy as np

from core.observation_builder.base_builder import BaseObservationBuilder

logger = logging.getLogger(__name__)


class ObservationBuilder1Inp(BaseObservationBuilder):
    def __init__(self, context, features):
        self.context = context
        self.features = features

    def reset(self):
        """Сброс параметров"""
        for feat in self.features:
            feat.reset()

    def get(self):
        data = [feat.get() for feat in self.static]
        observation = [np.array(data, dtype=np.float32)]
        return observation
