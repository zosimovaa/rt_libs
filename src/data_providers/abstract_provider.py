"""
Модуль содержит базовый класс AbstractDataProvider, используемые для загрузки задасетов


Для обучения и для торгов данные нужны в разбивке по периодам.
В таком случае приоритетным считается подход, где в качестве входных параметров выступают периоды
start - начальный периолд 
end - конечный период
period - собственно, периол

------------------------------------
Период принимаем обозначать по началу

start_ts - начало выгрузки. Совпадает со start
end_ts - окончание выгрузки. Равно end+period-1

Если мы запрашиваем данные в торговле, то
ts - текущий таймстемп. Отталкиваетс|""""""""""""""""""""я от него во всем

end = ts - period + 1
start = end - period * num_of_periods

end_ts = end + period - 1
start_ts = start
------------------------------------
Период принимаем обозначать по окончанию

start_ts - начало выгрузки. Совпадает со start - period + 1
end_ts - окончание выгрузки. Равно end

Если мы запрашиваем данные в торговле, то
ts - текущий таймстемп.

end = ts
start = end - period * num_of_periods - period + 1

end_ts = end 
start_ts = end - period + 1


"""

import pytz
import datetime
import pandas as pd


class AbstractDataProvider:
    """Базовый класс, который объявляет интерфейс DataProvider.
    Не содержит реализаци подключения к данным, но предоставляет базовые методы для пред и пост обработки данных.
    """
    def get_by_time(self, start, end, period, pair) -> pd.DataFrame:
        """
        Метод возвращает данные по котировкам. В параметрах задается начало и окончание периода.

        :param start: Начало периода. Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param end: Окончание периода. Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param period: Шаг выгрузки данных в секундах (60-300-600и т.д.)
        :param pair: Пара для отбора. Если None - будут овзвращены все пары.
        :return: pandas DataFrame с данными о котировках.
        """
        ts = self.date_to_unix_ts_in_utc(end)
        start_ts = self.date_to_unix_ts_in_utc(start)
        num_of_period = (ts - start_ts) // period

        result = self.get(ts, period, num_of_period, pair)
        return result

    def get(self, ts, period, num_of_period, pair) -> pd.DataFrame:
        """
        Метод возвращает данные по котировкам. В параметрах задается окончание периода и количество периодов 'в глубину'.
        :param ts: Окончание периода (крайняя дата). Может быть передано как в строковом формате, так и в UNIX TIMESTAMP.
        :param period: Шаг выгрузки данных в секундах (60-300-600и т.д.)
        :param num_of_period: количество шагов выгрузки данных.
        :param pair: Пар для отбора. Если None - будут овзвращены все пары.
        :return: pandas DataFrame с данными о котировках.
        """
        raise NotImplementedError

    @staticmethod
    def date_to_unix_ts_in_utc(date) -> int:
        """Метод преобразует время в строковом формате в UNIXTIMESTAMP в часовом поясе UTC"""
        if isinstance(date, str):
            timezone = pytz.timezone("UTC")
            without_timezone = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            with_timezone = timezone.localize(without_timezone)
            transformed = int(with_timezone.timestamp())
        else:
            transformed = int(date)
        return transformed

    @staticmethod
    def unix_ts_to_date(unix_ts):
        """Метод преобразует время в UNIXTIMESTAMP в строковый формат в часовом поясе UTC"""
        return datetime.datetime.utcfromtimestamp(int(unix_ts)).strftime('%Y-%m-%d %H:%M:%S.%f')

