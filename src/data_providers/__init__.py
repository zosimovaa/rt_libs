"""
data_providers - реализует доступ к данным по котировкам.
Сейчас есть 2 вида провайдеров, которые тянут инфо с биржи или из БД

Базовые метод доступа к данным - по периодам (кейс с трейдингом)
Т.е. для выборки данных указываем ts, период, кол-во периодов.

"""
from .clickhouse_connector import ClickHouseConnector
from .batch_task_handler import BatchTaskHandler
from .abstract_provider import AbstractDataProvider
from .dp_db_rt5 import DbDataProviderRT5
from .errors import DataProviderError, TooManyGapsError, UpToDateError


