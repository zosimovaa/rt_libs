import unittest
import numpy as np
import pandas as pd

from data_point import DataPointFactory, DataPointFactoryError


class DataPointFactoryTestCase(unittest.TestCase):

    @staticmethod
    def make_dataset(num=30, step=60, lowest_ask=96, highest_bid=95, ts_start=1000):

        lowest_asks = np.arange(lowest_ask, lowest_ask + step * num, step).reshape(-1, 1)
        highest_bids = np.arange(highest_bid, highest_bid + step * num, step).reshape(-1, 1)
        ts = np.arange(ts_start, ts_start + num * step, step).reshape(-1, 1)

        dataset = np.concatenate([ts, lowest_asks, highest_bids], axis=1)
        dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid"])
        dataset["ts"] = dataset["ts"].astype(int)
        dataset = dataset.set_index("ts")
        return dataset

    def check_lists(self, expected, actual):
        for i in range(len(expected)):
            self.assertEqual(expected[i], actual[i])

        self.assertEqual(len(expected), len(actual))

    def test_init(self):
        step_size = 60
        period = 300
        dataset = self.make_dataset(num=100, step=step_size, ts_start=1000)
        dpf = DataPointFactory(dataset, period=period, n_observation_points=5, n_future_points=3, step_size=step_size)

        self.assertEqual(max(dataset.index), dpf.max_step)

    def test_step_with_fp(self):
        step_size = 60
        period = 300
        dataset = self.make_dataset(num=100, step=step_size, ts_start=1000)
        n_observation_points = 5
        n_future_points = 3
        n_points = n_observation_points + n_future_points
        dpf = DataPointFactory(dataset, period=period, n_observation_points=n_observation_points, n_future_points=n_future_points, step_size=step_size)

        actual = dpf.get_idx()
        expected = np.arange(min(dpf.dataset.index), min(dpf.dataset.index) + n_points * period, period)

        self.check_lists(expected, actual)

    def test_step_without_fp(self):
        step_size = 60
        period = 300
        dataset = self.make_dataset(num=100, step=step_size, ts_start=1000)
        n_observation_points = 5
        n_future_points = 0
        n_points = n_observation_points + n_future_points
        dpf = DataPointFactory(dataset, period=period, n_observation_points=n_observation_points,
                               n_future_points=n_future_points, step_size=step_size)

        actual = dpf.get_idx()
        expected = np.arange(min(dpf.dataset.index), min(dpf.dataset.index) + n_points * period, period)

        self.check_lists(expected, actual)

    def test_step_same_step(self):
        step_size = 30
        period = 30
        dataset = self.make_dataset(num=100, step=step_size, ts_start=1000)
        n_observation_points = 5
        n_future_points = 5
        n_points = n_observation_points + n_future_points
        dpf = DataPointFactory(dataset, period=period, n_observation_points=n_observation_points,
                               n_future_points=n_future_points, step_size=step_size)

        actual = dpf.get_idx()
        expected = np.arange(min(dpf.dataset.index), min(dpf.dataset.index) + n_points * period, period)

        self.check_lists(expected, actual)


    def test_done_condition(self):
        step_size = 30
        period = 30
        dataset = self.make_dataset(num=100, step=step_size, ts_start=1000)
        n_observation_points = 5
        n_future_points = 5
        n_points = n_observation_points + n_future_points
        dpf = DataPointFactory(dataset, period=period, n_observation_points=n_observation_points,
                               n_future_points=n_future_points, step_size=step_size)

        done = False
        i = 0
        while not done:
            dp, done = dpf.get_next_step()
            i = i + 1

        expected = max(dpf.dataset.index)
        self.assertEqual(expected, dpf.cursor)

        dp, done = dpf.get_next_step()
        expected = max(dpf.dataset.index)
        self.assertEqual(expected, dpf.cursor)

if __name__ == '__main__':
    unittest.main()
