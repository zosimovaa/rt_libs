import unittest
import numpy as np
import pandas as pd
from collections import deque

from tests.test_suite_env import SimpleTestSuiteEnv
from core.context import BasicContext
from core.observation_builder.features.volume_features import TradeBalanceFeature


def make_dataset():
    lowest_asks = np.array([10, 20, 30, 40, 50, 60, 100, 200, 300, 400]).reshape(-1, 1)
    highest_bids = np.array([10, 20, 30, 40, 50, 60, 100, 200, 300, 400]).reshape(-1, 1)
    buy_vol = np.array([0, 0, 0, 1, 2, 1, 1.23, 100, 1,  0.00001]).reshape(-1, 1)
    sell_vol = np.array([0, 0, 0, 1, 1, 2, 1.23, 1, 100, 0.00002]).reshape(-1, 1)

    ts = np.arange(1000, 1000 + len(highest_bids), 1).reshape(-1, 1)

    dataset = np.concatenate([ts, lowest_asks, highest_bids, buy_vol, sell_vol], axis=1)
    dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid", "buy_vol", "sell_vol"])
    dataset["ts"] = dataset["ts"].astype(int)
    dataset = dataset.set_index("ts")
    return dataset


class TradeStateFeatureTestCase(unittest.TestCase):

    def check_lists(self, expected, actual):
        for i in range(len(expected)):
            np.testing.assert_almost_equal(np.round(expected[i], 2), np.round(actual[i], 2))
        self.assertEqual(len(expected), len(actual))

    def test_get(self):
        context = BasicContext()
        dataset = make_dataset()

        test_env = SimpleTestSuiteEnv(context, dataset)
        feature = TradeBalanceFeature(context)

        # Step 1 - zero volumes
        observed = feature.get()
        expected = [0, 0, 0]
        self.check_lists(expected, observed)

        # Step 2 - equal volumes
        test_env.next_step()
        observed = feature.get()
        expected = [0, 0, 0]
        self.check_lists(expected, observed)

        # Step 3 - 2x difference
        test_env.next_step()
        observed = feature.get()
        expected = [0, 0, 0.333]
        self.check_lists(expected, observed)

        # Step 4 - 2x difference opposite
        test_env.next_step()
        observed = feature.get()
        expected = [0, 0.333, -0.333]
        self.check_lists(expected, observed)

        # Step 5 - equal volumes
        test_env.next_step()
        observed = feature.get()
        expected = [0.333, -0.333, 0]
        self.check_lists(expected, observed)

        # Step 6 - equal volumes
        test_env.next_step()
        observed = feature.get()
        expected = [-0.333, 0, 0.98]
        self.check_lists(expected, observed)

        # Step 7 - equal volumes
        test_env.next_step()
        observed = feature.get()
        expected = [0, 0.98, -0.98]
        self.check_lists(expected, observed)

        # Step 8 - small volumes
        test_env.next_step()
        observed = feature.get()
        expected = [0.98, -0.98, -0.33]
        self.check_lists(expected, observed)


if __name__ == '__main__':
    unittest.main()
