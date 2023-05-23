"""
Рефакторинг DataPointFactory
Цель - запилить фабрику, работающую с любым шагом датасета

"""

from .data_point import DataPoint
import numpy as np
import logging
from basic_application import with_exception

logger = logging.getLogger(__name__)


class DataPointFactoryError(Exception):
    pass


class DataPointFactory:
    def __init__(self, dataset, step_size=1, offset=10, observation_len=10, future_points=0, alias=None):
        """
        Все параметры указываются в единицах, без привязки к шагу датасета
        :param dataset: pandas dataframe с данными
        :param step_size: шаг по датасету. Указывается в единицах
        :param offest: 'хвост' с историческими данными - количество точек, которые надо захватить.
        Рассчитывается как произведение n_observation на max scale_factor
        :param observation_len: количество точек в наблюдении
        :param future_points: количество точек в будущем
        """
        self.dataset = dataset
        self.offset = offset
        self.step_size = step_size
        self.observation_len = observation_len
        self.future_points = future_points
        self.done = False
        self.alias = alias

        # self.period = self.dataset.index[1] - self.dataset.index[0]
        self.max_cursor = self.dataset.shape[0] - self.future_points - 1 # -1 - т.к. индексация с нуля
        self.max_steps = self.max_cursor - self.offset + 1
        self.cursor = self.offset

        self.reset()

    def reset(self):
        """Сброс в начальное состояние"""
        self.done = False

        # Курсор устанавливается на значение, которое отстоит от начала так, чтобы от начала
        # до курсора был размер данных для одного датапоинта.
        self.cursor = self.offset  # min(self.dataset.index) + self.period * (self.offset - 1)
        data_point = self.get_current_step()
        return data_point

    def get_current_step(self):
        """ Возвращает текущий data_point"""
        data = self.dataset.iloc[self.cursor - self.offset : self.cursor + self.future_points , : ]

        data_point = DataPoint(
            data,
            future_points=self.future_points,
            observation_len=self.observation_len
        )
        return data_point

    def get_next_step(self):
        """Переводит курсор на величину шага и возвращает data_point"""
        if not self.done:
            self.cursor = self.cursor + self.step_size

        if self.cursor >= self.max_cursor:
            self.done = True

        data_point = self.get_current_step()
        return data_point, self.done

    def get_max_steps(self):
        return self.max_steps
