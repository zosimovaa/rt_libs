import numpy as np
import pandas as pd

import unittest
from data_point import DataPoint


class MainDataPointTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MainDataPointTestCase, self).__init__(*args, **kwargs)

        self.dataset = self.make_dataset()
        self.n_future_points = 3

    @staticmethod
    def make_dataset():
        lowest_asks = np.array([99, 100, 102, 105, 109, 114, 120, 127]).reshape(-1, 1)
        highest_bids = np.array([99.5, 100.5, 102.5, 105.5, 109.5, 114.5, 120.5, 127.5]).reshape(-1, 1)
        ts = np.arange(1000, 1000 + len(highest_bids) * 300, 300).reshape(-1, 1)
        dataset = np.concatenate([ts, lowest_asks, highest_bids], axis=1)
        dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid"])
        dataset["ts"] = dataset["ts"].astype(int)
        dataset = dataset.set_index("ts")
        return dataset

    def test_current_index(self):
        """Проверяем корректность расчета текущего индекса"""
        n_future_points = 3
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        value = dp.data.index[-n_future_points - 1]
        current_index = dp.get_current_ts()
        self.assertEqual(current_index, value)

        n_future_points = 0
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        value = dp.data.index[-n_future_points - 1]
        current_index = dp.get_current_ts()
        self.assertEqual(current_index, value)

    def test_get_timestamps(self):
        n_future_points = 3
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        timestamps = dp.get_timestamps()
        for i in range(dp.obs_len):
            self.assertEqual(timestamps[i], dp.data.index[i])
        actual = len(dp.data.index[:i + 1])
        expected = len(timestamps)
        self.assertEqual(expected, actual)

        n_future_points = 0
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        timestamps = dp.get_timestamps()
        for i in range(dp.obs_len):
            self.assertEqual(timestamps[i], dp.data.index[i])
        actual = len(dp.data.index[:i + 1])
        expected = len(timestamps)
        self.assertEqual(expected, actual)

    def test_get_value(self):
        n_future_points = 3
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        col_name = "lowest_ask"
        idx = dp.get_current_ts()
        expected = self.dataset.loc[idx, col_name]
        actual = dp.get_value(col_name)
        self.assertEqual(expected, actual)

        n_future_points = 0
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        col_name = "lowest_ask"
        idx = dp.get_current_ts()
        expected = self.dataset.loc[idx, col_name]
        actual = dp.get_value(col_name)
        self.assertEqual(expected, actual)

        n_future_points = 2
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        col_name = "lowest_ask"
        cursor = dp.data.index[2]
        expected = self.dataset.loc[cursor, col_name]
        actual = dp.get_value(col_name, cursor=cursor)
        self.assertEqual(expected, actual)

    def test_get_values(self):
        n_future_points = 3
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        col_name = "lowest_ask"
        cursor = dp.get_current_ts()

        actual = dp.get_values(col_name)
        expected = self.dataset.loc[:cursor, col_name]

        for i in range(len(expected)):
            self.assertEqual(expected.iloc[i], actual.iloc[i])

        self.assertEqual(len(expected), len(actual))

    def test_get_last_diffs(self):
        n_future_points = 3
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        col_name = "lowest_ask"
        num = 3

        actual = dp.get_last_diffs(num, col_name)

        expected = []
        idxs = dp.get_timestamps()
        for i in range(num):
            expected.append(self.dataset.loc[idxs[-i-1], col_name] - self.dataset.loc[idxs[-i-2], col_name])

        expected = expected[::-1]
        for i in range(len(expected)):
            self.assertEqual(expected[i], actual[i])

        self.assertEqual(len(expected), len(actual))

    def test_get_future_values(self):
        n_future_points = 4
        dp = DataPoint(self.dataset, n_future_points=n_future_points)
        col_name = "lowest_ask"

        actual = dp.get_future_values(col_name)
        cursor = dp.get_current_ts() + dp.period

        expected = self.dataset.loc[cursor:, col_name]

        for i in range(len(expected)):
            self.assertEqual(expected.iloc[i], actual.iloc[i])
        self.assertEqual(len(expected), len(actual))

if __name__ == '__main__':
    unittest.main()
