"""
Базовый билдер с двумя фичами:
 - состояние сделки
 - данные для CNN
   - курс
   - профит

Данные для сети подаются в формате [num_of_points, num_features]
        rate perp        profit repr
array([[-1.9651896e-01,  0.0000000e+00],
       [-1.8928000e-01,  0.0000000e+00],
       [-1.8809754e-01,  0.0000000e+00]]
"""

import logging
import numpy as np

from core.observation_builder.base_builder import BaseObservationBuilder

logger = logging.getLogger(__name__)


class ObservationBuilder2Inp(BaseObservationBuilder):
    def __init__(self, context, static=[], cnn=[]):
        self.context = context
        self.static_features = static
        self.cnn_features = cnn

    def reset(self):
        """Сброс параметров"""
        for feat in self.static_features:
            feat.reset()

        for feat in self.cnn_features:
            feat.reset()

    def get(self):
        static_data = [feat.get() for feat in self.static_features]
        series_data = [feat.get() for feat in self.cnn_features]
        series_data = np.concatenate(series_data, axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(series_data, dtype=np.float32)
        ]

        return observation


        return observation
