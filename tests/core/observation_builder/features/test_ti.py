import unittest
import numpy as np
import pandas as pd
from collections import deque

from tests.test_suite_env import SimpleTestSuiteEnv
from core.context import BasicContext
from core.observation_builder.features.trend_indicator import TrendIndicatorFeature


def make_dataset():
    lowest_asks = np.array([10, 20, 30, 40, 50, 60, 100, 200, 300, 400]).reshape(-1, 1)
    highest_bids = np.array([10, 20, 30, 40, 50, 60, 100, 200, 300, 400]).reshape(-1, 1)
    ts = np.arange(1000, 1000 + len(highest_bids), 1).reshape(-1, 1)

    dataset = np.concatenate([ts, lowest_asks, highest_bids], axis=1)
    dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid"])
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

        test_env = SimpleTestSuiteEnv(context, dataset, n_future_points=3)
        feature = TrendIndicatorFeature(context)

        # Step 1 - check default trade status
        observed = feature.get()
        expected = [177.78, 88.888,  59.26]
        self.check_lists(expected, observed)

        # Step 2
        test_env.next_step()
        observed = feature.get()
        expected = [88.888, 59.26, 61.11]
        self.check_lists(expected, observed)

        # Step 3
        test_env.open_trade()
        observed = feature.get()
        expected = [88.888, 59.26, 61.11]
        self.check_lists(expected, observed)

        # Step 2
        test_env.next_step()
        observed = feature.get()
        expected = [59.26, 61.11, 108.89]
        self.check_lists(expected, observed)

if __name__ == '__main__':
    unittest.main()
