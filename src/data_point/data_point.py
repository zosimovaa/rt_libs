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


class DataPoint:
    """DataPoint, базовая версия"""
    def __init__(self, data, data_future=None):
        self.data = data
        self.data_f = data_future
        self.current_index = max(self.data.index)
        self.offset = len(data)

    def get_current_ts(self):
        """
        Возвращает текущее значение ts
        :return: int, текущее значение ts
        """
        return self.current_index

    def get_timestamps(self):
        """
        Возвращает все значения ts
        :return: numpy.ndarray, список всех ts
        """
        return self.data.index.values

    def get_value(self, name, cursor=None):
        """
        Целевой метод для получения выбранного значения из данных
        :param name: название колонки (lowest_ask, highest_bid или иное)
        :param cursor: значение индекса (не обязательный параметр)
        :return: значение заданной колонки
        """
        if cursor is None:
            cursor = self.current_index
        val = self.data.loc[cursor, name]
        return val

    def get_values(self, name):
        """
        Возвращает все значения данных инструмента (за все ts)
        :param name: название колонки (lowest_ask, highest_bid или иное)
        :return: pandas.z_core.series.Series, значения заданной колонки
        """
        data = self.data.loc[:, name]
        return data

    def get_current_data(self):
        """
        Возвращает весь датасет
        :return: pandas.z_core.frame.DataFrame
        """
        return self.data

    def get_last_diffs(self, num, column='lowest_ask'):
        """
        Метод возвращает разницу между значениями одной колонки, начиная с "конца", по индексу
        Т.е. от актуального значения.

        :param num: Количество возвращаемых точек данных
        :param column: название колонки
        :return: numpy.ndarray. Последнее значение соответствует разности последнего и предпоследнего значений в data
        """
        col_idx = np.argmax(self.data.columns == column)
        diffs = self.data.diff(axis=0).iloc[-num:, col_idx]
        return diffs.values

    def get_future_values(self, name):
        """
        Возвращает все будущние значения данных инструмента (за все ts) из заданной колонки.
        :param name: название колонки (lowest_ask, highest_bid или иное)
        :return: значение заданной цены
        """
        data = self.data_f.loc[:, name]
        return data

    def get_future_data(self):
        """
        Возвращает весь датасет будущих данных
        :return: pandas.z_core.frame.DataFrame
        """
        return self.data_f

