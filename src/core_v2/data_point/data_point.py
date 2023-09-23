"""
Класс DataPoint реализован, чтобы передать текущее представление точки данных в core_v2.

Для обычных фичей достаточно методов get_value и get_values со стандартной длиной выборки, равной observation_len
В некоторых случаях, возможно, потребуется получать весь хвост данных, чтобы взять объективное среднее для нормализации.

Дополнительные методы типа get_current_index, и get_value со специфическим индексом нужны в фиче с историей.
И то - для текущей точки достаточно current_index, а в init'е - наверное, можно использовать относительное указание индекса

Т.к. пока нет необходимости работать с историей (сделал быстро вычисление на лету) - то достаточно двух базовых методов

Второй поинт - пока нет работы с future_points - отрицательную индексацию модно исключить.
Это должно повысить производлительность, но, возможно, игра не стоит свеч и можно все оставить в одном классе, надо тестить.


Из этой реализации убраны некоторые методы, которые были в старой в силу их ненадобности.'
"""
import numpy as np

from basic_application import with_exception


class DataPointError(Exception):
    pass


class DataPoint:
    """DataPoint, описание смотри в модуле."""

    def __init__(self, data, columns, indexes, observation_len=5, future_points=0):
        self.data = data
        self.columns = columns
        self.indexes = indexes

        # Длина наблюдения. Для разных step_factor включает в себя разное количество точек данных.
        self.observation_len = observation_len
        # Количество точек данных в будущем
        self.future_points = future_points
        # Количество точек данных, доступных для построения observation с учетом всех доступных step_factors
        self.tail_points = self.data.shape[0] - self.future_points

        # В "абстрактных" тестовых сценариях присутствуют кейсы, где дата поинт состоит из одной точки.
        # На синтетических и реальных данных такого нет, observation_len всегда больше 1
        if len(self.indexes) > 1:
            self.period = self.indexes[1] - self.indexes[0]
        else:
            self.period = 1

        # Верхушка observation = текущая точка данных.
        # Так это последняя точка данных, но future_points может ее сместить ближе к началу.
        self.cursor = self.tail_points - 1  # Курсор про номер позии

    def _get_num(self, step_factor, num):
        "Вычисляет количество точек наблюдения и возволяет в большей части случаев исключить эту логику из потребителя"
        if num is None:
            # Базовый сценарий
            return self.observation_len
        elif num == -1:
            # Вычисление позволяет передать максимальное количество точек для вычисления value со step_factor выше 1.
            return self.tail_points // (self.observation_len * step_factor) * (self.observation_len * step_factor)
        else:
            return num

    def get_index_slice(self, step_factor=1, num=None):
        """Возвращает все курсоры, в том числе расположенные между реперными точками"""
        num = self._get_num(step_factor, num)

        # верхний индекс на единицу выше, т.к. это элемент не захватывается при слайсинге
        up_bound = self.cursor + 1
        low_bound = up_bound - num * step_factor

        return low_bound, up_bound

    def get_values(self, name, step_factor=1, num=None, agg="average"):
        """Возвращает значений точек данных заданной длины и агрегированное для step_factor выше 1
        """
        num = self._get_num(step_factor, num)
        low_bound, up_bound = self.get_index_slice(step_factor=step_factor, num=num)

        col_idx = self.columns.tolist().index(name)
        values = self.data[low_bound: up_bound, col_idx]
        if step_factor > 1:
            # оптимизация производительности
            values = getattr(np, agg)(values.reshape(-1, step_factor), axis=1)
        return values

    def get_value(self, name, step_factor=1, agg="average"):
        """Метод возвращает одно значение агрегированное для step_factor выше 1"""
        col_idx = self.columns.tolist().index(name)
        value = self.data[-1 * step_factor, col_idx]
        if step_factor > 1:
            value = np.mean(value)
        return value

    def get_current_cursor(self):
        return self.cursor

    def get_current_index(self):
        return self.indexes[self.cursor]

    def get_points(self, step_factor=1, num=None):
        """Возвращает индексы по реперным точкам - т.е. будет соответствовать количеству запрошенных точек"""
        num = self._get_num(step_factor, num)

        if num >= 0:
            up_bound = self.cursor + 1
            low_bound = up_bound - (num - 1) * step_factor - 1
        else:
            low_bound = self.cursor + step_factor
            up_bound = low_bound - num * step_factor

        idxs = self.indexes[low_bound: up_bound: step_factor]
        return idxs