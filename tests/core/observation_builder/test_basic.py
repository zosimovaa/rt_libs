import unittest
import numpy as np
import pandas as pd
from collections import deque

from data_point import DataPointFactory, DataPointFactoryError
from core.observation_builder.basic import ObservationBuilderBasic
from core.context import BasicContext
from core.actions import TradeAction


class ObservationBuilderBasicTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ObservationBuilderBasicTestCase, self).__init__(*args, **kwargs)

        self.dpf = None
        self.dataset = None
        self.context = None
        self.ob = None
        self.n_observation_points = None

        self.expected_profit = None

    def init_env(self, n_observation_points=4, n_future_points=0):
        self.n_observation_points = n_observation_points
        self.dataset = self.make_dataset()
        self.dpf = DataPointFactory(
            self.dataset,
            period=1,
            n_observation_points=n_observation_points,
            n_future_points=n_future_points
        )
        self.context = BasicContext(market_fee=0.)
        self.ob = ObservationBuilderBasic(self.context)
        self.expected_profit = deque(np.zeros(self.n_observation_points), maxlen=self.n_observation_points)

    @staticmethod
    def make_dataset():
        lowest_asks = np.array([98, 99, 99, 100, 101, 102, 103, 104]).reshape(-1, 1)
        highest_bids = np.array([99, 99, 100, 101, 102, 103, 104, 105]).reshape(-1, 1)
        ts = np.arange(1000, 1000 + len(highest_bids), 1).reshape(-1, 1)

        dataset = np.concatenate([ts, lowest_asks, highest_bids], axis=1)
        dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid"])
        dataset["ts"] = dataset["ts"].astype(int)
        dataset = dataset.set_index("ts")
        return dataset

    def check_lists(self, expected, actual):
        for i in range(len(expected)):
            np.testing.assert_almost_equal(np.round(expected[i], 2), np.round(actual[i], 2))
        self.assertEqual(len(expected), len(actual))

    def check_rate(self, obs, data_point):
        actual = obs[1][:, 0]
        current_val = data_point.get_value("highest_bid")
        vals = data_point.get_values("highest_bid").values
        expected = (vals / current_val - 1) * self.ob.SCALE_FACTOR
        self.check_lists(expected, actual)

    def check_profit(self, obs, trade):
        profit_point = trade.get_profit() * self.ob.SCALE_FACTOR
        self.expected_profit.append(profit_point)
        actual = obs[1][:, 1]
        self.check_lists(self.expected_profit, actual)

    def test_trade_status(self):
        # Init test environment
        self.init_env()

        # First step
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.assertEqual(0, obs[0])

        # Second step
        trade = TradeAction(self.context)
        self.context.set_trade(trade)

        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.assertEqual(1, obs[0])

        # Third step
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.assertEqual(1, obs[0])

        # Fourth step
        trade.close()
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.assertEqual(0, obs[0])

    def test_rate_repr(self):
        # Init test environment
        self.init_env()

        # First step
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.check_rate(obs, data_point)

        # Second step
        trade = TradeAction(self.context)
        self.context.set_trade(trade)
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.check_rate(obs, data_point)

        # Third step
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.check_rate(obs, data_point)

        # Fourth step
        trade.close()
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.check_rate(obs, data_point)

    def test_profit_repr(self):
        # Init test environment
        self.init_env()

        # First step
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)

        actual = obs[1][:, 1]
        self.check_lists(self.expected_profit, actual)

        # Second step
        trade = TradeAction(self.context)
        self.context.set_trade(trade)
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.check_profit(obs, trade)

        # Third step
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.check_profit(obs, trade)

        # Fourth step
        trade.close()
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        obs = self.ob.get(data_point)
        self.check_profit(obs, trade)


if __name__ == '__main__':
    unittest.main()
