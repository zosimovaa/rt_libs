"""
Модуль содержит реализацию классов, которые описывают результат выполнения действия на бирже.
Инстансы впоследствии используются :
 - для сбора метрики эпизода обучения
 - для построения графика работы сети на наборе данных (в плеере)
 - для накопления статистики торгов в БД
"""

import uuid


class BaseAction:
    """Базовый класс действия"""
    def __init__(self):
        self.id = str(uuid.uuid4())


class BadAction(BaseAction):
    """Класс описывает неверное действие нейросети в текущем контексте"""
    def __init__(self, ts, action):
        BaseAction.__init__(self)
        self.ts = ts
        self.action = action


class TradeAction(BaseAction):
    """Класс описывает торговую операцию"""
    def __init__(self, ts, lowest_ask, highest_bid, market_fee):
        """Инстанс класса создается при открытии торговой операции"""
        BaseAction.__init__(self)
        self.open_ts = ts
        self.open_price = lowest_ask
        self.close_ts = None
        self.close_price = None
        self.is_open = True
        self.last_price = highest_bid
        self.profit = self.get_profit()
        self.trade_volume = 0
        self.market_fee = market_fee

    def update(self, highest_bid):
        if self.is_open:
            self.last_price = highest_bid
            self.profit = self.get_profit()

    def close(self, ts, highest_bid):
        if self.is_open:
            self.close_ts = ts
            self.close_price = highest_bid
            self.last_price = highest_bid
            self.profit = self.get_profit()
            self.is_open = False

    def get_profit(self):
        if self.is_open:
            profit = self.last_price / self.open_price - 1 - self.market_fee
        else:
            profit = 0
        return profit
