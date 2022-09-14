import unittest
import numpy as np
import pandas as pd
from collections import deque

from tests.test_suite_env import SimpleTestSuiteEnv
from core.context import BasicContext
from core.observation_builder.features.basic_features import TradeStateFeature, Rates1DFeature, ProfitFeature


def make_dataset():
    lowest_asks = np.array([10, 20, 40, 160, 320, 640, 1280]).reshape(-1, 1)
    highest_bids = np.array([10, 20, 40, 160, 320, 640, 1280]).reshape(-1, 1)
    ts = np.arange(1000, 1000 + len(highest_bids), 1).reshape(-1, 1)

    dataset = np.concatenate([ts, lowest_asks, highest_bids], axis=1)
    dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid"])
    dataset["ts"] = dataset["ts"].astype(int)
    dataset = dataset.set_index("ts")
    return dataset


class TradeStateFeatureTestCase(unittest.TestCase):
    def test_get(self):
        context = BasicContext()
        dataset = make_dataset()

        test_env = SimpleTestSuiteEnv(context, dataset)
        feature = TradeStateFeature(context)

        # Step 1 - check default trade status
        observed = feature.get()
        expected = False
        self.assertEqual(expected, observed)

        # Step 2 - check no changes with new step
        test_env.next_step()
        observed = feature.get()
        expected = False
        self.assertEqual(expected, observed)

        # Step 3 - open trade
        test_env.open_trade()
        observed = feature.get()
        expected = True
        self.assertEqual(expected, observed)

        # Step 4 - check no changes with new step
        test_env.next_step()
        observed = feature.get()
        expected = True
        self.assertEqual(expected, observed)

        # Step 5 - close trade
        test_env.close_trade()
        observed = feature.get()
        expected = False
        self.assertEqual(expected, observed)

        # Step 6 - check no changes with new step
        test_env.next_step()
        observed = feature.get()
        expected = False
        self.assertEqual(expected, observed)


class Rates1DFeatureTestCase(unittest.TestCase):
    def check_lists(self, expected, actual):
        for i in range(len(expected)):
            np.testing.assert_almost_equal(np.round(expected[i], 2), np.round(actual[i], 2))
        self.assertEqual(len(expected), len(actual))

    def test_get(self):
        context = BasicContext()
        dataset = make_dataset()
        test_env = SimpleTestSuiteEnv(context, dataset)
        feature = Rates1DFeature(context)
        feature.SCALE_FACTOR = 1

        # Step 1 - first step
        observed = feature.get()
        expected = [-0.75, -0.5,  0. ]
        self.check_lists(expected, observed)

        # Step 2 - next step rate representation
        test_env.next_step()
        observed = feature.get()
        expected = [-0.875, -0.75, 0.]
        self.check_lists(expected, observed)

        # Step 3 - check no changes with new step
        test_env.open_trade()
        observed = feature.get()
        expected = [-0.875, -0.75, 0.]
        self.check_lists(expected, observed)


class ProfitFeatureTestCase(unittest.TestCase):
    def check_lists(self, expected, actual):
        for i in range(len(expected)):
            np.testing.assert_almost_equal(np.round(expected[i], 2), np.round(actual[i], 2))
        self.assertEqual(len(expected), len(actual))

    def test_get(self):
        context = BasicContext()
        dataset = make_dataset()
        test_env = SimpleTestSuiteEnv(context, dataset)

        self.expected_profit = deque(np.zeros(test_env.n_observation_points), maxlen=test_env.n_observation_points)

        feature = ProfitFeature(context)
        feature.SCALE_FACTOR = 1

        # Step 1 - default value
        observed = feature.get()
        expected = [0., 0., 0.]
        self.check_lists(expected, observed)

        # Step 2 - check no changes with new step
        test_env.next_step()
        observed = feature.get()
        expected = [0., 0., 0.]
        self.check_lists(expected, observed)

        # Step 3 - open trade. With current logic profit at this step equals 0!
        test_env.open_trade()
        observed = feature.get()
        expected = [0., 0., 0.]
        self.check_lists(expected, observed)

        # Step 4
        test_env.next_step()
        observed = feature.get()
        expected = [0., 0., 1.]
        self.check_lists(expected, observed)

        # Step 5 -
        test_env.next_step()
        observed = feature.get()
        expected = [0., 1., 3.]
        self.check_lists(expected, observed)

        # Step 6 -
        test_env.next_step()
        observed = feature.get()
        expected = [1., 3., 7.]
        self.check_lists(expected, observed)

        # Step 7 -
        test_env.close_trade()
        observed = feature.get()
        expected = [0., 0., 0.]
        self.check_lists(expected, observed)

        # Step 8 -
        test_env.next_step()
        observed = feature.get()
        expected = [0., 0., 0.]
        self.check_lists(expected, observed)

if __name__ == '__main__':
    unittest.main()