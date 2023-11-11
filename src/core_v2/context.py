"""
Синглотон контекст

"""


class Singleton(type):
    """Metaclass for train and trade contexts"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        alias = kwargs["alias"]
        if alias not in cls._instances:
            cls._instances[alias] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[alias]


class Context(metaclass=Singleton):
    """Context for train and trade pipelines"""
    STORED_PARAMS = ("lowest_ask", "highest_bid")

    def __init__(self, alias="TRAIN"):
        self.alias = alias
        self.params = {}

    def reset(self):
        self.params = {}

    def get(self, name):
        return self.params.get(name)

    def put(self, name, value):
        self.params[name] = value

    def set_dp(self, data_point):
        # пункты 1, 2 , 4 по отдельности и все вместе длятся одинаково = 1300 ms.
        # проблема в извлечении stored params

        # 1. Set data_point 1300ms
        self.put("data_point", data_point)

        # 2. Set ts 1300ms
        ts = data_point.get_index()
        self.put("ts", ts)

        # 3. Extract stored params 4300ms
        for key in self.STORED_PARAMS:
            try:
                value = data_point.get_value(key)
                self.put(key, value)
            except:
                pass

        # 4. Update profit 1300ms
        if self.get("is_open"):
            trade = self.get("trade")
            price = self.get("highest_bid")
            profit = trade.get_profit(price)
        else:
            profit = 0
        self.put("profit", profit)


class ContextConsumer:
    """Базовый класс потребителя контекста, дает доступ к контекту"""
    def __init__(self, alias):
        self.alias = alias
        self.context = Context(alias=alias)
