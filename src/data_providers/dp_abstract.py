"""
Модуль содержит базовые классы DataProvider, используемые для загрузки задасетов
"""

import pytz
import datetime
import pandas as pd


class DataProviderError(Exception):
    """Класс ошибки DataProvider. Необходим при обработке ошибок в трейдере """
    pass


class AbstractDataProvider:
    """Базовый класс, который обхявляет интерфейс DataProvider.
    Не содержит реализаци подключения к данным, но предоставляет базовые методы для пред и пост обработки данных.
    """

    def get(self, start, end, period, pairs=[]) -> pd.DataFrame:
        """
        Метод возвращает данные по котировкам. В параметрах задается начало и окончание периода.

        :param start: Начало периода. Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param end: Окончание периода. Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param period: Шаг выгрузки данных в секундах (60-300-600и т.д.)
        :param pairs: Список пар для отбора. Если список пустой - будут овзвращены все пары.
        :return: pandas DataFrame с данными о котировках.
        """
        raise NotImplementedError

    def get_by_periods(self, ts, period, num_of_period, pairs=[]) -> pd.DataFrame:
        """
        Метод возвращает данные по котировкам. В параметрах задается окончание периода и количество периодов 'в глубину'.
        :param ts: Окончание периода (крайняя дата). Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param period: Шаг выгрузки данных в секундах (60-300-600и т.д.)
        :param num_of_period: количество шагов выгрузки данных.
        :param pairs: Список пар для отбора. Если список пустой - будут овзвращены все пары.
        :return: pandas DataFrame с данными о котировках.
        """
        start = ts - period * num_of_period
        end = ts
        result = self.get(start, end, period, pairs=pairs)
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
