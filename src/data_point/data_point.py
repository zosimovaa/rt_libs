
"""
Класс DataPoint реализован, чтобы передать текущее представление точки данных в core.

Данный класс решает несколько проблем:
 1. На этапе обучение позвозяет передать дополнительные данные о точках в будущем
 для расчета индикатора тренда. В целевом решении эта задача будет решаться с помощью дополнительной модели.

 2.Содержит в себе необходимые методы для получения нужных представлений данных.

Основные тезисы:
 - Индексы реальные, не обнуляются (не начинаются с нуля).
 - Текущее значение имеет индекс с максимальным значением, остальные значения считаются историчсекими.
 - Информация о будущих точках представлена в отдельном поле и не пересекается с датасетом.
 - Реализация DataPoint отвязана от названий полей, чтобы была возможность работать с разными датасетами.

"""
import numpy as np

from basic_application import with_exception


class DataPointError(Exception):
    pass


class DataPoint:
    def __init__(self, data, future_points=0, observation_len=5):
        self.data = data
        self.offset = future_points
        self.observation_len = observation_len

        # Верхушка observation = текущая точка данных
        #print(f"data.shape[0] - {data.shape[0]}")
        #print(f"self.offset - {self.offset}")
        self.cursor = data.shape[0] - self.offset - 1
        #print(f"self.cursor - {self.cursor}")

        #print(f"data.index.values - {data.index.values}")

        self.current_idx = data.index.values[self.cursor]


        if observation_len > 1:
            self.period = data.index[1] - data.index[0]
        else:
            self.period = 1

    def get_points(self, step_factor=1, num=None):
        """Возвращает индексы по реперным точкам - т.е. будет соответствовать количеству запрошенных точек"""
        if num is None:
            num = self.observation_len

        if num >= 0:
            up_bound = self.cursor + 1
            low_bound = up_bound - (num - 1) * step_factor - 1
        else:
            low_bound = self.cursor + step_factor
            up_bound = low_bound - num * step_factor

        idxs = self.data.index.values[low_bound: up_bound: step_factor]
        return idxs

    def get_indexes(self, step_factor=1, num=None):
        """Возвращает все индексы, в том числе расположенные между реперными точками"""
        if num is None:
            num = self.observation_len

        if num >= 0:
            up_bound = self.cursor + 1
            low_bound = up_bound - num * step_factor

        else:
            low_bound = self.cursor + 1
            up_bound = low_bound - num * step_factor

        # idxs = self.data.index.values[low_bound : up_bound : 1]
        # return idxs
        return low_bound, up_bound

    def get_values(self, name, step_factor=1, num=None):
        if num is None:
            num = self.observation_len

        low_bound, up_bound = self.get_indexes(step_factor=step_factor, num=num)
        col = self.data.loc[:, name].values
        result = col[low_bound: up_bound]

        return result

    def get_value(self, name, step_factor=1, idx=None):
        """Метод возвращает одно значение. По умолчанию для текущего индекса.
        Индекс можно задать из вне. Данные возвращаются с учетом scale_factor
        Расчет по одной точке (_build_point в AbstractFeatureWithHistory) для scalefactor>1 будет неточным -
        там надо считать все сразу. Надо исследовать, сходу сложно оценить.
        """
        if idx is None:
            idx = self.current_idx

        cursor = np.where(self.data.index.values == idx)[0][0]
        col = self.data.loc[:, name].values

        result = col[cursor - step_factor + 1: cursor + 1]
        return result

    def get_current_index(self):
        return self.current_idx
