"""Модуль реализует тестирование класса BadAction
2 отличия от теста основнойторговой операции
 - Профит считается с нулевым marketfee и все завязано на highest_bid
   (в обычной торговой операци цена открыти - lowest_ask)

 - состояние сделки берется из самой торговой операции а не из признака в контексте.
   В торговой операции он обновляется в онлайне, а в контексте - после datapoint_update
"""

import unittest
import numpy as np
import collections

from tests.test_dataset import TestDatasetGenerator

from src.data_point import DataPointFactory
from src.core_v1.context import BasicContext
from src.core_v1.actions import OppositeTradeAction
from src.core_v1.observation_builder.features import OppositeProfitFeature


class OppositeProfitTestCase(unittest.TestCase):
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
        profit_feat = OppositeProfitFeature(self.context, scale_factor=1)

        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        profit_feat.reset()

        # Step 1 - Проверка профита при ресете
        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 2 - новая точка данных
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 3 - новая точка данных + торговая операция
        opposite_trade = OppositeTradeAction(self.context)
        self.context.set("trade", opposite_trade, domain="OppositeTrade")

        open_price = self.context.get("highest_bid")

        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 4 - первая тока при открытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        observed = profit_feat.get()
        current_price = self.context.get("highest_bid")
        current_profit = current_price / open_price - 1

        self.expected.append(current_profit)
        self.check_profit(observed)

        # Step 5 - вторая тока при открытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        observed = profit_feat.get()
        current_price = self.context.get("highest_bid")
        current_profit = current_price / open_price - 1
        self.expected.append(current_profit)
        self.check_profit(observed)

        # Step 6 - закрываем позицию
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        # Т.к. контектс обновляется только при апдейте датапоинта, то после закрытия и до обновления датапоинта
        # мы все еще видим открытую транзакцию и рассчитываем профит. Это не есть хорошо.
        # p.s. нужно менять архитектуру и переходить на эвент бас

        opposite_trade.close()
        self.expected = collections.deque(np.zeros(self.n_observation_points), maxlen=self.n_observation_points)
        observed = profit_feat.get()
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
        opposite_trade = OppositeTradeAction(self.context)
        self.context.set("trade", opposite_trade, domain="OppositeTrade")
        open_price = self.context.get("highest_bid")

        observed = profit_feat.get()
        self.check_profit(observed)

        # Step 10(4) - первая тока при открытой позиции
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        observed = profit_feat.get()
        current_price = self.context.get("highest_bid")
        current_profit = current_price / open_price - 1
        self.expected.append(current_profit)
        self.check_profit(observed)


if __name__ == '__main__':
    unittest.main()
