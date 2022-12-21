"""
Класс DataPoint реализован, чтобы передать текущее представление точки данных в z_core.

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
    def __init__(self, data, n_observation_points=10, n_future_points=0, period=1):
        self.data = data

        self.fut_len = n_future_points
        self.obs_len = n_observation_points
        self.hist_len = len(self.data) - n_future_points - n_observation_points

        # Верхушка observation = текущая точка данных
        self.current_idx = self.data.index[-(self.fut_len + 1)]

        # Начало observation
        self.start_idx = self.data.index[self.hist_len]

        self.period = period  # self.data.index[1] - self.data.index[0]

    def _get_slice(self, num):
        if num >= 0:
            end_idx = self.data.shape[0] - self.fut_len
            start_idx = self.data.shape[0] - num - self.fut_len
        else:
            end_idx = self.data.shape[0] - self.fut_len - num
            start_idx = self.data.shape[0] - self.fut_len
        return start_idx, end_idx

    def get_values(self, name, num=None):
        if num is None:
            num = self.obs_len
        start_idx, end_idx = self._get_slice(num)
        result = self.data.loc[:, name].values[start_idx : end_idx]
        return result

    def get_value(self, name, cursor=None):
        idx = self.get_indexes(cursor=cursor, num=1)
        result = self.data.at[idx[0], name]
        return result

    def get_current_index(self):
        return self.current_idx

    def get_indexes(self, cursor=None, num=None):
        if cursor is None:
            cursor = self.current_idx

        if num is None:
            num = self.obs_len

        period = self.period

        if num > 0:
            stop = cursor - (period * (num - 1))
            idxs = np.arange(stop, cursor + period, period)
        else:
            stop = cursor - (period * (num - 1))
            idxs = np.arange(cursor + period, stop, period)

        return idxs
