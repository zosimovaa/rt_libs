"""
Модуль содержит базовые классы DataProvider, используемые для загрузки задасетов
"""

import pytz
import datetime
import pandas as pd


class DataProviderError(Exception):
    """Класс ошибки DataProvider. Необходим при обработке ошибок в трейдере """
    pass


class TooManyGapsError(DataProviderError):
    MESSAGE = "{0} gaps in {1} records"
    """Класс ошибки DataProvider. Необходим при обработке ошибок в трейдере """
    def __init__(self, gaps, total):
        self.message = self.MESSAGE.format(gaps, total)
        self.gaps = gaps
        self.total = total
        super().__init__(self.message)

    def __str__(self):
        return self.message


class UpToDateError(DataProviderError):
    """Класс ошибки DataProvider. Необходим при обработке ошибок в трейдере """
    pass


class AbstractDataProvider:
    """Базовый класс, который обхявляет интерфейс DataProvider.
    Не содержит реализаци подключения к данным, но предоставляет базовые методы для пред и пост обработки данных.
    """
    def get(self, start, end, period, pair=None) -> pd.DataFrame:
        """
        Метод возвращает данные по котировкам. В параметрах задается начало и окончание периода.

        :param start: Начало периода. Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param end: Окончание периода. Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param period: Шаг выгрузки данных в секундах (60-300-600и т.д.)
        :param pair: Пара для отбора. Если None - будут овзвращены все пары.
        :return: pandas DataFrame с данными о котировках.
        """
        raise NotImplementedError

    def get_by_periods(self, ts, period, num_of_period, pair=None) -> pd.DataFrame:
        """
        Метод возвращает данные по котировкам. В параметрах задается окончание периода и количество периодов 'в глубину'.
        :param ts: Окончание периода (крайняя дата). Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param period: Шаг выгрузки данных в секундах (60-300-600и т.д.)
        :param num_of_period: количество шагов выгрузки данных.
        :param pair: Пар для отбора. Если None - будут овзвращены все пары.
        :return: pandas DataFrame с данными о котировках.
        """
        ts_end = self.date_to_unix_ts_in_utc(ts)
        start = ts_end - period * num_of_period
        end = ts_end
        result = self.get(start, end, period, pair=pair)
        return result

    @staticmethod
    def date_to_unix_ts_in_utc(date) -> int:
        """Метод преобразует время в строковом формате в UNIXTIMESTAMP.
        """
        if isinstance(date, str):
            timezone = pytz.timezone("UTC")
            without_timezone = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            with_timezone = timezone.localize(without_timezone)
            transformed = int(with_timezone.timestamp())
        else:
            transformed = int(date)
        return transformed

