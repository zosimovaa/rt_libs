"""
Класс DataPoint реализован, чтобы передать текущее представление точки данных в core_v3.

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

    def __init__(self, data, columns, indexes, observation_len=5, future_points=0, step_size=1):
        self.data = data
        self.columns = columns
        self.indexes = indexes
        self.step_size = step_size  # шаг по датасету, нужно как дефолтное значение периода

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

    def _get_data_len(self, period, num):
        """Вычисляет количество точек наблюдения, необходимое для формирования ответа"""
        if period is None:
            period = self.step_size

        if period % self.period > 0:
            raise Exception(f"Bad period - {period}. Data period is  {self.period}")

        period_len = int(period / self.period)

        if num is None:
            # Базовый сценарий
            data_len = self.observation_len * period_len

        elif num == -1:
            # Вычисление позволяет передать максимальное количество точек для вычисления value со step_factor выше 1.
            data_len = self.tail_points // (self.observation_len * period_len) * self.observation_len
        else:
            data_len = num * period_len

        return data_len, period_len

    def get_index_slice(self, period=None, num=None):
        """Возвращает все индексы, в том числе расположенные между реперными точками"""
        data_len, period_len = self._get_data_len(period, num)
        # верхний индекс на единицу выше, т.к. это элемент не захватывается при слайсинге
        up_bound = self.cursor + 1
        low_bound = up_bound - data_len
        return low_bound, up_bound, period_len

    def get_values(self, name, period=None, num=None, agg="average"):
        """Возвращает значения точек данных заданной длины и агрегированное для period"""
        if period is None:
            period = self.step_size
        low_bound, up_bound, period_len = self.get_index_slice(period=period, num=num)
        col_idx = self.columns.tolist().index(name)
        values = self.data[low_bound: up_bound, col_idx]
        if period_len > 1:
            # оптимизация производительности
            values = getattr(np, agg)(values.reshape(-1, period_len), axis=1)
        return values

    def get_indexes(self, period=None, num=None):
        """Возвращает индексы по реперным точкам - т.е. не возвращает промежуточные значения"""
        if period is None:
            period = self.step_size
        low_bound, up_bound, period_len = self.get_index_slice(period=period, num=num)
        values = self.indexes[low_bound + period_len - 1: up_bound + period_len: period_len]
        return values

    def get_value(self, name, period=None, agg="average"):
        """Метод возвращает одно значение агрегированное для period выше 1"""
        value = self.get_values(name, period=period, num=1, agg=agg)
        return value[0]

    def get_index(self):
        """Метод возвращает текущее значение индекса"""
        return self.indexes[self.cursor]

    def get_cursor(self):
        """Метод возвращает текущее значение курсора - абсолютное значенеи положения курсора в массиве"""
        return self.cursor
