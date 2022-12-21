"""Модуль реализует тестирование класса BadAction"""

import unittest
import numpy as np

from tests.test_dataset import TestDatasetGenerator

from src.data_point import DataPointFactory
from src.core.context import BasicContext
from src.core.actions import TradeAction
from src.core.observation_builder.features import Rates1DFeature


class Rates1DTestCase(unittest.TestCase):
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
        scale_factor = 5

        rates_feat = Rates1DFeature(self.context, scale_factor=scale_factor)
        rates_feat.reset()

        observed = rates_feat.get()
        expected = np.array(self.dataset.loc[:, "highest_bid"].values[:self.n_observation_points])
        expected = expected / expected[-1] - 1
        expected = expected * scale_factor

        for i in range(len(expected)):
            self.assertEqual(np.round(observed[i][0], 3), np.round(expected[i], 3))


        # ==== Step 2 ====

        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        observed = rates_feat.get()
        expected = np.array(self.dataset.loc[:, "highest_bid"].values[1:self.n_observation_points+1])
        expected = expected / expected[-1] - 1
        expected = expected * scale_factor



        for i in range(len(expected)):
            self.assertEqual(np.round(observed[i][0], 3), np.round(expected[i], 3))

if __name__ == '__main__':
    unittest.main()
