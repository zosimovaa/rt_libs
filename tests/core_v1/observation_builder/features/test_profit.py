"""Модуль реализует тестирование класса BadAction"""

import unittest
import numpy as np
import collections

from tests.test_dataset import TestDatasetGenerator

from src.data_point import DataPointFactory
from src.core_v1.context import BasicContext
from src.core_v1.actions import TradeAction
from src.core_v1.observation_builder.features import ProfitFeature


class ProfitTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_observation_points = 6
        self.market_fee = 0.005
        self.dataset = TestDatasetGenerator().make(
            num=100,
            lowest_ask=100,
            highest_bid=99,
            ts_start=1000,
            price_step=1,
            period=60
        )
        self.dpf = DataPointFactory(
            self.dataset,
            period=60,
            n_observation_points=self.n_observation_points,
            n_history_points=0,
            n_future_points=0,
            step_size=None
        )
        self.context = BasicContext(market_fee=self.market_fee)
        self.expected = collections.deque(np.zeros(self.n_observation_points), maxlen=self.n_observation_points)

    def check_profit(self, observed):
        for i in range(len(observed)):
            self.assertEqual(np.round(observed[i][0], 5), np.round(self.expected[i], 5))

    def test_main_flow(self):
        profit_feat = ProfitFeature(self.context, scale_factor=1)

        # Step 1 - Проверка профита при ресете
        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        profit_feat.reset()
        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 2 - новая точка данных
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 3 - новая точка данных + торговая операция
        trade_action = TradeAction(self.context)
        self.context.set_trade(trade_action)
        open_price = self.context.get("lowest_ask")

        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 4 - первая тока при открытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        observed = profit_feat.get()
        current_price = self.context.get("highest_bid")
        current_profit = current_price/open_price - 1 - self.context.market_fee
        self.expected.append(current_profit)
        self.check_profit(observed)

        # Step 5 - вторая тока при открытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        observed = profit_feat.get()
        current_price = self.context.get("highest_bid")
        current_profit = current_price / open_price - 1 - self.context.market_fee
        self.expected.append(current_profit)
        self.check_profit(observed)

        # Step 6 - закрываем позицию
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        # Т.к. контектс обновляется только при апдейте датапоинта, то после закрытия и до обновления датапоинта
        # мы все еще видим открытую транзакцию и рассчитываем профит. Это не есть хорошо.
        # p.s. нужно менять архитектуру и переходить на эвент бас

        trade_action.close()
        observed = profit_feat.get()

        current_price = self.context.get("highest_bid")
        current_profit = current_price / open_price - 1 - self.context.market_fee
        self.expected.append(current_profit)
        self.check_profit(observed)

        # Step 7 - тест при закрытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        self.expected = collections.deque(np.zeros(self.n_observation_points), maxlen=self.n_observation_points)
        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 8 - тест при закрытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        self.expected = collections.deque(np.zeros(self.n_observation_points), maxlen=self.n_observation_points)
        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 9(3) - новая точка данных + торговая операция
        trade_action = TradeAction(self.context)
        self.context.set_trade(trade_action)
        open_price = self.context.get("lowest_ask")

        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 10(4) - первая тока при открытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        observed = profit_feat.get()
        current_price = self.context.get("highest_bid")
        current_profit = current_price / open_price - 1 - self.context.market_fee
        self.expected.append(current_profit)
        self.check_profit(observed)


if __name__ == '__main__':
    unittest.main()
