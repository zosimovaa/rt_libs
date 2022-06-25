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
    def __init__(self, data, n_future_points=0):
        self.data = data
        self.current_index = self.data.index[-(n_future_points + 1)]
        self.obs_len = len(data) - n_future_points
        self.fut_len = n_future_points
        self.period = self.data.index[1] - self.data.index[0]

    @with_exception(DataPointError)
    def get_current_ts(self):
        return self.current_index

    @with_exception(DataPointError)
    def get_timestamps(self):
        if self.fut_len:
            return self.data.index[:-self.fut_len].values
        else:
            return self.data.index.values

    @with_exception(DataPointError)
    def get_value(self, name, cursor=None):
        if cursor is None:
            cursor = self.current_index
        val = self.data.loc[cursor, name]
        return val

    @with_exception(DataPointError)
    def get_values(self, name):
        data = self.data.loc[:self.current_index, name]
        return data

    @with_exception(DataPointError)
    def get_last_diffs(self, num, column='lowest_ask'):
        row_idx_start = self.get_timestamps()[-num-1]
        row_idx_end = self.current_index

        data = self.data.loc[row_idx_start : row_idx_end, column]
        diffs = data.diff()
        return diffs.values[1:]

    @with_exception(DataPointError)
    def get_current_data(self):
        return self.data.loc[:self.current_index + 1]

    @with_exception(DataPointError)
    def get_future_values(self, name, cursor=None):
        if self.fut_len:
            cursor = self.current_index if cursor is None else cursor
            start_idx = cursor + self.period
            end_idx = self.fut_len * self.period + cursor
            data = self.data.loc[start_idx: end_idx, name]
        else:
            data = None
        return data

    @with_exception(DataPointError)
    def get_future_data(self, cursor=None):
        """Возвращает все фичи для текущей точки, коотрые считаются 'будущими' """
        if self.fut_len:
            cursor = self.current_index if cursor is None else cursor
            start_idx = cursor + self.period
            end_idx = self.fut_len * self.period + cursor
            data = self.data.loc[start_idx: end_idx, :]
        else:
            data = None
        return data
