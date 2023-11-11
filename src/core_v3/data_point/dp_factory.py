"""
Рефакторинг DataPointFactory
Цель - запилить фабрику, работающую с любым шагом датасета

"""

from .data_point import DataPoint
import logging

logger = logging.getLogger(__name__)


class DataPointFactory:
    """DataPointFactory из core v3 работает с периодом"""
    def __init__(self, dataset, step_size=1, offset=10, observation_len=10, future_points=0, alias=None):
        """
        Все параметры указываются в единицах, без привязки к шагу датасета
        :param dataset: pandas dataframe с данными
        :param step_size: шаг по датасету. Указывается в единицах индекса.
        :param offest: 'хвост' с историческими данными - количество точек, которые надо захватить,
        чтобы сформировать observation для любого period.
        :param observation_len: количество точек в наблюдении
        :param future_points: количество точек в будущем
        """

        self.data = dataset.values              # Данные
        self.columns = dataset.columns          # Названия колонок
        self.indexes = dataset.index            # Индексы (время)

        self.observation_len = observation_len  # Длина наблюдения. Не может быть  больше offset
        self.future_points = future_points      # Количество точек из будущего, которые захватим (стоят правее курсора)
        self.offset = offset                    # Исторический хвост, который всегда прицепляем к текущему курсору
        self.cursor = offset                    # Текущий шаг по датасету. Начальное значение равно оффсету
        self.step_size = step_size              # Размер шага по датасету (в единицах индекса)

        if len(self.indexes) > 1:
            self.data_period = self.indexes[1] - self.indexes[0]
            if step_size % self.data_period != 0:
                raise Exception(f"Data period ({self.data_period}) is not multiplicity to Step size ({step_size}). Take another step size value")

        else:
            self.data_period = 1

        self.cursor_step_size = int(step_size / self.data_period)  # Шаг по датасету.

        self.done = False                       # Признак достижения конца датасета
        self.alias = alias                      # Возможно, это рудимент, который уже можно удалить

        self.max_cursor = self.data.shape[0] - self.future_points - 1  # -1 - т.к. индексация с нуля
        self.max_steps = (self.max_cursor - self.offset) // self.cursor_step_size + 1

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
        low_bound = self.cursor - self.offset
        up_bound = self.cursor + self.future_points

        data = self.data[low_bound: up_bound, :]
        indexes = self.indexes.values[low_bound: up_bound]

        data_point = DataPoint(
            data,
            self.columns,
            indexes,
            future_points=self.future_points,
            observation_len=self.observation_len,
            step_size=self.step_size
        )
        return data_point

    def get_next_step(self):
        """Переводит курсор на величину шага и возвращает data_point"""
        if not self.done:
            self.cursor = min(self.cursor + self.cursor_step_size, self.max_cursor)

            if self.cursor == self.max_cursor:
                self.done = True

        data_point = self.get_current_step()
        return data_point, self.done

    def get_max_steps(self):
        return self.max_steps
