"""Модуль реализует тестирование класса BadAction"""

import unittest

from tests.test_dataset import TestDatasetGenerator

from core_v1.data_point import DataPointFactory
from src.core_v1.context import BasicContext
from src.core_v1.actions import TradeAction, OppositeTradeAction


class TradeActionTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_observation_points = 4
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

    def test_main_flow(self):
        # ==== Step 1 ====
        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        self.trade_action = TradeAction(self.context)
        self.context.set_trade(self.trade_action)
        # check trade
        self.assertEqual(self.trade_action.market_fee, self.market_fee)  # add assertion here
        self.assertEqual(self.trade_action.is_open, True)

        open_price = self.context.get("lowest_ask")

        # ==== Step 2 ====
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        # check profit
        current_price = self.context.get("highest_bid")
        profit_expected = round(current_price / open_price - 1 - self.market_fee, 5)
        profit_observed = self.trade_action.get_profit()
        self.assertEqual(profit_observed, profit_expected)  # add assertion here
        self.assertEqual(self.trade_action.is_open, True)

        # ==== Step 3 ====
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        # check profit
        current_price = self.context.get("highest_bid")
        profit_expected =  round(current_price/open_price  - 1 - self.market_fee, 5)
        profit_observed = self.trade_action.get_profit()
        self.assertEqual(profit_observed, profit_expected)  # add assertion here
        self.assertEqual(self.trade_action.is_open, True)


        # ==== Step 4 ====
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        self.trade_action.close()

        # check profit
        current_price = self.context.get("highest_bid")
        profit_expected = round(current_price / open_price - 1 - self.market_fee, 5)
        profit_observed = self.trade_action.get_profit()
        self.assertEqual(profit_observed, profit_expected)  # add assertion here
        self.assertEqual(self.trade_action.is_open, False)

    def test_main_flow_opposite(self):
        # ==== Step 1 ====
        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        self.trade_action = OppositeTradeAction(self.context)
        self.context.set_trade(self.trade_action)
        # check trade
        self.assertEqual(self.trade_action.market_fee, 0)  # add assertion here
        self.assertEqual(self.trade_action.is_open, True)

        open_price = self.context.get("highest_bid")

        # ==== Step 2 ====
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        # check profit
        current_price = self.context.get("highest_bid")
        profit_expected = round(current_price / open_price - 1, 5)
        profit_observed = self.trade_action.get_profit()
        self.assertEqual(profit_observed, profit_expected)  # add assertion here
        self.assertEqual(self.trade_action.is_open, True)

        # ==== Step 3 ====
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        # check profit
        current_price = self.context.get("highest_bid")
        profit_expected =  round(current_price/open_price  - 1, 5)
        profit_observed = self.trade_action.get_profit()
        self.assertEqual(profit_observed, profit_expected)  # add assertion here
        self.assertEqual(self.trade_action.is_open, True)


        # ==== Step 4 ====
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        self.trade_action.close()

        # check profit
        current_price = self.context.get("highest_bid")
        profit_expected = round(current_price / open_price - 1, 5)
        profit_observed = self.trade_action.get_profit()
        self.assertEqual(profit_observed, profit_expected)  # add assertion here
        self.assertEqual(self.trade_action.is_open, False)

if __name__ == '__main__':
    unittest.main()
