"""Модуль реализует тестирование класса BadAction"""

import unittest

from tests.test_dataset import TestDatasetGenerator

from core_v1.data_point import DataPointFactory
from src.core_v1.context import BasicContext
from src.core_v1.actions import TradeAction
from src.core_v1.observation_builder.features import TradeStateFeature


class TradeStateTestCase(unittest.TestCase):
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

        trade_state_feat = TradeStateFeature(self.context)
        trade_state_feat.reset()

        observed = trade_state_feat.get()
        expected = False
        self.assertEqual(observed, expected)

        # ==== Step 2 ====
        trade_action = TradeAction(self.context)
        self.context.set_trade(trade_action)
        # check trade
        observed = trade_state_feat.get()
        expected = True
        self.assertEqual(observed, expected)

        # ==== Step 3 ====
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        # check trade
        observed = trade_state_feat.get()
        expected = True
        self.assertEqual(observed, expected)

        # ==== Step 4 ====
        trade_action.close()
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        # check trade
        observed = trade_state_feat.get()
        expected = False
        self.assertEqual(observed, expected)


if __name__ == '__main__':
    unittest.main()
