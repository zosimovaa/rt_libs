import numpy as np
from collections import deque


class AbstractFeature:
    """Для простых, не тяжелых в вычислении фичей"""

    def __init__(self, context):
        self.context = context

    def get(self):
        """Метод возвращает текущее значение признака для данного datapoint"""
        pass

    def reset(self):
        """Сброс в начальное состояние. Актуально для наследников AbstractFeatureWithHistory"""
        pass


class AbstractFeatureWithHistory(AbstractFeature):
    """Для тяжелых в вычислении фичей. Для каждого шага вычисляется только актуальная точка данных. Предыдущие хранятся в кэше"""

    def __init__(self, context):
        self.context = context
        self.data = None
        self.last_update = None

    def get(self):
        """Определяет логику формирования данных"""
        dp = self.context.data_point

        # Запрос первого наблюдения
        if self.data is None:
            self.data = self._init_data(dp)
            self.last_update = dp.get_current_ts()

        # Были потери в данных
        elif dp.get_current_ts() > self.last_update + dp.period:
            self.data = self._init_data(dp)
            self.last_update = dp.get_current_ts()

        # Повторное построение observation
        elif dp.get_current_ts() == self.last_update:
            pass

        # Запрос следующей точки
        else:
            point = self._build_point(dp)
            self.data.append(point)
            self.last_update = dp.get_current_ts()

        return np.array(self.data)

    def reset(self):
        self.data = None
        self.last_update = None

    def _build_point(self, dp, cursor=None):
        pass

    def _init_data(self, dp):
        data = deque(maxlen=len(dp.get_timestamps()))
        indexes = dp.get_timestamps()
        for cursor in indexes:
            point = self._build_point(dp, cursor=cursor)
            data.append(point)
        return data
