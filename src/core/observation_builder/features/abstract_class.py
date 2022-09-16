import abc
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class AbstractFeature(metaclass=abc.ABCMeta):
    """Для простых, не тяжелых в вычислении фичей"""

    def __init__(self, context):
        self.context = context
        self.scale_is_broken = False

    def get(self, norm=False, clip_edge=4) -> np.ndarray:
        data = self._get()

        # Нормирование данных
        if norm:
            data = data / np.abs(data).mean()

        # Проверка масштаба
        if clip_edge > 0:
            check_data = np.abs(data) > clip_edge
            if check_data.sum() > 0:
                self.scale_is_broken = True
                data = np.clip(data, -clip_edge, clip_edge)
        return data

    def reset(self) -> None:
        """Сброс в начальное состояние. Актуально для наследников AbstractFeatureWithHistory"""
        if self.scale_is_broken:
            logger.warning("{0}: Scale is broken".format(self.__class__.__name__))
            self.scale_is_broken = False
        pass

    @abc.abstractmethod
    def _get(self) -> np.ndarray:
        """Метод возвращает текущее значение признака для данного datapoint"""


class AbstractFeatureWithHistory(AbstractFeature):
    """
    Для тяжелых в вычислении фичей. Для каждого шага вычисляется только актуальная точка данных. Предыдущие хранятся в кэше.
    Но лучше использовать precompute-фичи для предварительного расчета в датасете, если это возможно.
    """

    def __init__(self, context):
        super().__init__(context)
        self.context = context
        self.data = None
        self.last_update = None

    @abc.abstractmethod
    def _build_point(self, dp, cursor=None):
        """Метод должен реализовывать вычисление точки данных для данного cursor. Если cursor не определн - то """
        pass

    def _get(self):
        """Определяет логику формирования данных"""
        dp = self.context.data_point

        # Запрос первого наблюдения
        if self.data is None:
            self.data = self._init_data(dp)
            self.last_update = dp.get_current_index()

        # Были потери в данных
        elif dp.get_current_index() > self.last_update + dp.period:
            self.data = self._init_data(dp)
            self.last_update = dp.get_current_index()

        # Повторное построение observation
        elif dp.get_current_index() == self.last_update:
            pass

        # Запрос следующей точки
        else:
            point = self._build_point(dp)
            self.data.append(point)
            self.last_update = dp.get_current_index()

        data = np.array(self.data)

        return data

    def reset(self):
        super().reset()
        self.data = None
        self.last_update = None

    def _init_data(self, dp):
        indexes = dp.get_indexes()
        data = deque(maxlen=len(indexes))
        for cursor in indexes:
            point = self._build_point(dp, cursor=cursor)
            data.append(point)
        return data
