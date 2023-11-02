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

    def _get_num(self, period, num):
        """Вычисляет количество точек наблюдения, необходимое для формирования ответа"""
        if period % self.period > 0:
            raise Exception(f"Bad period - {period}. Data period is  {self.period}")

        if num is None:
            # Базовый сценарий
            return int(self.observation_len * period/self.period)
        elif num == -1:
            # Вычисление позволяет передать максимальное количество точек для вычисления value со step_factor выше 1.
            return int(self.tail_points / (self.observation_len * period//self.period) * self.observation_len)
        else:
            return num

    def get_index_slice(self, period=1, num=None):
        """Возвращает все индексы, в том числе расположенные между реперными точками"""
        num = self._get_num(period, num)
        # верхний индекс на единицу выше, т.к. это элемент не захватывается при слайсинге
        up_bound = self.cursor + 1
        low_bound = up_bound - num
        return low_bound, up_bound

    def get_values(self, name, period=1, num=None, agg="average"):
        """Возвращает значения точек данных заданной длины и агрегированное для step_factor выше 1"""
        #num = self._get_num(period, num)   #здесь мы должны получить все имеющиеся точки данных "от сих до сих", чтобы потом агрегировать в запрошенные значения.
        low_bound, up_bound = self.get_index_slice(period=period, num=num)
        col_idx = self.columns.tolist().index(name)
        values = self.data[low_bound: up_bound, col_idx]
        if period/self.period > 1:
            # оптимизация производительности
            values = getattr(np, agg)(values.reshape(-1, int(period/self.period)), axis=1)
        return values

    def get_value(self, name, period=1, agg="average"):
        """Метод возвращает одно значение агрегированное для step_factor выше 1"""
        col_idx = self.columns.tolist().index(name)
        value = self.data[-1 * int(period/self.period), col_idx]
        if period/self.period > 1:
            value = np.mean(value)
        return value

    def get_current_cursor(self):
        return self.cursor

    def get_current_index(self):
        return self.indexes[self.cursor]

    def get_indexes(self, period=1, num=None):
        """Возвращает индексы по реперным точкам - т.е. будет соответствовать количеству запрошенных точек"""
        num = self._get_num(period, num)
        low_bound, up_bound = self.get_index_slice(period=period, num=num)
        values = self.indexes[low_bound: up_bound: int(period/self.period)]
        return values