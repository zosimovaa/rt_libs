import os
from clickhouse_driver import connect
import logging

logger = logging.getLogger(__name__)


class ClickHouseConnector:
    """Обертка в виде менеджера контекста для подключения к БД"""
    def __init__(self, params):
        if "user" not in params.keys():
            params["user"] = os.getenv("DB_USER")

        if "password" not in params.keys():
            params["password"] = os.getenv("DB_PASS")

        self.params = params
        self.conn = None
        self.cursor = None

    def create_connection(self):
        self.conn = connect(**self.params)
        self.cursor = self.conn.cursor()
        logger.warning("Cursor created, database connection established")

    def close_connection(self):
        self.cursor = None
        self.conn.close()
        logger.warning("Cursor closed")

    def __enter__(self):
        self.create_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()
