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
    def __init__(self, data, n_history_points=10, n_future_points=0):
        self.data = data

        self.fut_len = n_future_points
        self.hist_len = n_history_points
        self.obs_len = len(data) - n_future_points - n_history_points

        # Верхушка observation = текущая точка данных
        self.up_idx = self.data.index[-(n_future_points + 1)]

        # Начало observation
        self.low_idx = self.data.index[n_history_points]

        self.period = self.data.index[1] - self.data.index[0]

    # Работа с индексом = = = = = = = = = = = = = = = = = = = =
    @with_exception(DataPointError)
    def get_current_ts(self):
        return self.up_idx

    @with_exception(DataPointError)
    def get_timestamps(self):
        if self.fut_len:
            return self.data.index[self.hist_len: -self.fut_len].values
        else:
            return self.data.index[self.hist_len:].values

    # Получение точек данных = = = = = = = = = = = = = = = = = = = =
    @with_exception(DataPointError)
    def get_value(self, name, cursor=None):
        if cursor is None:
            cursor = self.up_idx
        val = self.data.loc[cursor, name]
        return val

    @with_exception(DataPointError)
    def get_values(self, name):
        data = self.data.loc[self.low_idx : self.up_idx, name]
        return data

    # todo перевести все реализации на такой метод
    @with_exception(DataPointError)
    def get_values2(self, name=None, cursor=None, num=1):
        """
        :param name: Название колонки. Если отсутствует - будет возвращен весь массив
        :param cursor: Индекс запрашиваемого элемента. При отсутствии будет возвращен актуальный.
        :param num: Количество элементов от запрашиваемого положительынй индекс - уход в историю,
        отрицательный - в будущие точки; 0, -1 и 1 - вернет текущий.
        :return: pandas DataFrame
        """
        if cursor is None:
            cursor = self.up_idx

        if name is None:
            name = self.data.columns

        if num < 0:
            return self.data.loc[cursor:, name].iloc[:-num]
        elif num == 0:
            return self.data.loc[cursor]
        else:
            return self.data.loc[:cursor, name].iloc[-num:]

    # todo вынести эту логику в ticker, чтобы datapoint был только с получением данных
    @with_exception(DataPointError)
    def get_last_diffs(self, num, column='lowest_ask'):
        row_idx_start = self.get_timestamps()[-num-1]
        row_idx_end = self.up_idx

        data = self.data.loc[row_idx_start : row_idx_end, column]
        diffs = data.diff()
        return diffs.values[1:]

    @with_exception(DataPointError)
    def get_current_data(self):
        return self.data.loc[self.low_idx  : self.up_idx + 1]

    @with_exception(DataPointError)
    def get_future_values(self, name, cursor=None):
        if self.fut_len:
            cursor = self.up_idx if cursor is None else cursor
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
            cursor = self.up_idx if cursor is None else cursor
            start_idx = cursor + self.period
            end_idx = self.fut_len * self.period + cursor
            data = self.data.loc[start_idx: end_idx, :]
        else:
            data = None
        return data

    @with_exception(DataPointError)
    def get_hist_values(self, name, cursor=None):
        if self.hist_len:
            cursor = self.low_idx if cursor is None else cursor

            start_idx = max(cursor - self.period * self.hist_len, min(self.data.index))
            end_idx = cursor - self.period

            data = self.data.loc[start_idx: end_idx, name]
        else:
            data = None
        return data

    @with_exception(DataPointError)
    def get_hist_data(self, cursor=None):
        """Возвращает все фичи для текущей точки, коотрые считаются 'будущими' """
        if self.hist_len:
            cursor = self.low_idx if cursor is None else cursor

            start_idx = max(cursor - self.period * self.hist_len, min(self.data.index))
            end_idx = cursor - self.period

            data = self.data.loc[start_idx: end_idx, :]
        else:
            data = None
        return data


class DataPoint2:
    def __init__(self, data, n_observation_points=10, n_future_points=0):
        self.data = data

        self.fut_len = n_future_points
        self.obs_len = n_observation_points
        self.hist_len = len(data) - n_future_points - n_observation_points

        # Верхушка observation = текущая точка данных
        self.current_idx = self.data.index[-(self.fut_len + 1)]

        # Начало observation
        self.start_idx = self.data.index[self.hist_len]

        self.period = self.data.index[1] - self.data.index[0]

    # Работа с индексом = = = = = = = = = = = = = = = = = = = =
    @with_exception(DataPointError)
    def get_current_ts(self):
        return self.current_idx

    @with_exception(DataPointError)
    def get_timestamps(self):
        if self.fut_len:
            return self.data.index[self.hist_len: -self.fut_len].values
        else:
            return self.data.index[self.hist_len:].values

    def get_ts(self, cursor=None, step_factor=None, num=None):
        if cursor is None:
            cursor = self.current_idx



    # todo перевести все реализации на такой метод
    @with_exception(DataPointError)
    def get_values(self, name=None, cursor=None, step_factor=1, num=1):
        """
        :param name: Название колонки. Если отсутствует - будет возвращен весь массив
        :param cursor: Индекс запрашиваемого элемента. При отсутствии будет возвращен актуальный.
        :param num: Количество элементов от запрашиваемого положительынй индекс - уход в историю,
        отрицательный - в будущие точки; 0, -1 и 1 - вернет текущий.
        :return: pandas DataFrame
        """
        if cursor is None:
            cursor = self.current_idx

        if name is None:
            name = self.data.columns

        if num < 0:
            return self.data.loc[cursor:, name].iloc[:-num]
        elif num == 0:
            return self.data.loc[cursor]
        else:
            return self.data.loc[:cursor, name].iloc[-num:]

    # Получение точек данных = = = = = = = = = = = = = = = = = = = =
    @with_exception(DataPointError)
    def get_value1(self, name, cursor=None):
        if cursor is None:
            cursor = self.current_idx
        val = self.data.loc[cursor, name]
        return val

    @with_exception(DataPointError)
    def get_values1(self, name):
        data = self.data.loc[self.start_idx: self.current_idx, name]
        return data



    # todo вынести эту логику в ticker, чтобы datapoint был только с получением данных
    @with_exception(DataPointError)
    def get_last_diffs(self, num, column='lowest_ask'):
        row_idx_start = self.get_timestamps()[-num-1]
        row_idx_end = self.current_idx

        data = self.data.loc[row_idx_start : row_idx_end, column]
        diffs = data.diff()
        return diffs.values[1:]

    @with_exception(DataPointError)
    def get_current_data(self):
        return self.data.loc[self.start_idx: self.current_idx + 1]

    @with_exception(DataPointError)
    def get_future_values(self, name, cursor=None):
        if self.fut_len:
            cursor = self.current_idx if cursor is None else cursor
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
            cursor = self.current_idx if cursor is None else cursor
            start_idx = cursor + self.period
            end_idx = self.fut_len * self.period + cursor
            data = self.data.loc[start_idx: end_idx, :]
        else:
            data = None
        return data

    @with_exception(DataPointError)
    def get_hist_values(self, name, cursor=None):
        if self.hist_len:
            cursor = self.start_idx if cursor is None else cursor

            start_idx = max(cursor - self.period * self.hist_len, min(self.data.index))
            end_idx = cursor - self.period

            data = self.data.loc[start_idx: end_idx, name]
        else:
            data = None
        return data

    @with_exception(DataPointError)
    def get_hist_data(self, cursor=None):
        """Возвращает все фичи для текущей точки, коотрые считаются 'будущими' """
        if self.hist_len:
            cursor = self.start_idx if cursor is None else cursor

            start_idx = max(cursor - self.period * self.hist_len, min(self.data.index))
            end_idx = cursor - self.period

            data = self.data.loc[start_idx: end_idx, :]
        else:
            data = None
        return data