from .....data_providers import ClickHouseConnector


class DbDataProvider:

    WHERE_PLACEHOLDER = "!WHERE!"

    def __init__(self, params, get_query, put_query):
        self.params = params
        self.get_query = get_query
        self.put_query = put_query
        self.put_errors = []

    def put(self, data):
        if self.params is not None:
            with ClickHouseConnector(self.params) as conn:
                try:
                    # todo сделать потом пост-обработку ошибок
                    conn.cursor.executemany(self.put_query, data + self.put_errors)
                except Exception as e:
                    print(e)
                    self.put_errors.append(data)

                    raise Exception("DB connect error") from e
                else:
                    print(f"Записано строк {conn.cursor.rowcount}")

    def get(self, **kwargs):
        if self.params is not None:
            get_query = self._build_filter(**kwargs)
            with ClickHouseConnector(self.params) as conn:
                try:
                    conn.cursor.execute(get_query, parameters=kwargs)
                    data = conn.cursor.fetchall()
                except Exception as e:
                    print(e)
                    raise Exception("DB connect error") from e
                else:
                    print(f"Прочитано строк {conn.cursor.rowcount}")
        return data
    def _build_filter(self, **kwargs):
        where = []
        for key in kwargs.keys():
            where.append(f"{key}=%({key})s")

        if len(where) > 1:
            where = " WHERE " + " AND ".join(where)
        elif len(where) == 1:
            where = " WHERE " + where[0]
        else:
            where = ""

        return self.get_query.replace(self.WHERE_PLACEHOLDER, where)
