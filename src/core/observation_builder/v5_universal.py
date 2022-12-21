"""
Базовый билдер с двумя фичами:
 - состояние сделки
 - данные для НС
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

from .interface import ObservationBuilderInterface

logger = logging.getLogger(__name__)


class ObservationBuilder2Dim(ObservationBuilderInterface):
    def __init__(self, context, static, series):
        """Конструктор класса"""
        self.context = context
        self.static = static
        self.series = series

    def reset(self):
        """Сброс параметров"""
        for feat in self.static:
            feat.reset()

        for feat in self.series:
            feat.reset()

    def get(self):
        static_data = [feat.get() for feat in self.static]
        series_data = [feat.get() for feat in self.series]
        series_data = np.concatenate(series_data, axis=1)

        observation = [
            np.array(static_data, dtype=np.float32),
            np.array(series_data, dtype=np.float32)
        ]

        return observation

