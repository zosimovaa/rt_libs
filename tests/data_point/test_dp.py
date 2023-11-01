import numpy as np
import pandas as pd

import unittest
from core_v1.data_point import DataPoint


class MainDataPointTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MainDataPointTestCase, self).__init__(*args, **kwargs)

        self.dataset = self.make_dataset()
        self.n_future_points = 1
        self.observation_len = 4
        self.scale_factor = (1, 2)

    @staticmethod
    def make_dataset():
        lowest_asks = np.array([99, 100, 102, 105, 109, 114, 120, 127, 130]).reshape(-1, 1)
        highest_bids = np.array([99.5, 100.5, 102.5, 105.5, 109.5, 114.5, 120.5, 127.5, 130.5]).reshape(-1, 1)
        ts = np.arange(1000, 1000 + len(highest_bids) * 300, 300).reshape(-1, 1)
        dataset = np.concatenate([ts, lowest_asks, highest_bids], axis=1)
        dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid"])
        dataset["ts"] = dataset["ts"].astype(int)
        dataset = dataset.set_index("ts")
        return dataset

    def test_internal_variables(self):
        """Проверяем , что все внутренние переменные встали корректно"""
        dp = DataPoint(self.dataset, future_points=self.n_future_points, observation_len=self.observation_len)

        self.assertEqual(dp.offset, self.n_future_points)
        self.assertEqual(dp.data.shape, self.dataset.shape)
        self.assertEqual(dp.observation_len, self.observation_len)

        self.assertEqual(dp.cursor, self.dataset.shape[0] - self.n_future_points - 1)
        self.assertEqual(dp.current_idx, self.dataset.index[-self.n_future_points - 1])
        self.assertEqual(dp.period, 300)

    def test_get_points(self):
        dp = DataPoint(self.dataset, future_points=self.n_future_points, observation_len=self.observation_len)

        for sf in self.scale_factor:
            values = dp.get_points(step_factor=sf)

            up_bound = self.dataset.shape[0] - self.n_future_points + 1
            low_bound = up_bound - (self.observation_len - 1) * sf
            expected = self.dataset.index.values[low_bound : up_bound : sf]

            print(values)
            print(expected)

            lenght = len(values)
            for i in range(lenght):
                self.assertEqual(values[i], expected[i])


    def get_value(self):
        n_future_points = 3
        dp = DataPoint(self.dataset, future_points=n_future_points)
        col_name = "lowest_ask"
        idx = dp.get_current_index()
        expected = self.dataset.loc[idx, col_name]
        actual = dp.get_value(col_name)
        self.assertEqual(expected, actual)

        n_future_points = 0
        dp = DataPoint(self.dataset, future_points=n_future_points)
        col_name = "lowest_ask"
        idx = dp.get_current_index()
        expected = self.dataset.loc[idx, col_name]
        actual = dp.get_value(col_name)
        self.assertEqual(expected, actual)

        n_future_points = 2
        dp = DataPoint(self.dataset, future_points=n_future_points)
        col_name = "lowest_ask"
        cursor = dp.data.index[2]
        expected = self.dataset.loc[cursor, col_name]
        actual = dp.get_value(col_name, cursor=cursor)
        self.assertEqual(expected, actual)

    def get_values(self):
        n_future_points = 3
        dp = DataPoint(self.dataset, future_points=n_future_points)
        col_name = "lowest_ask"
        cursor = dp.get_current_index()

        actual = dp.get_values(col_name)
        expected = self.dataset.loc[:cursor, col_name]

        for i in range(len(expected)):
            self.assertEqual(expected.iloc[i], actual.iloc[i])

        self.assertEqual(len(expected), len(actual))



    def get_future_values(self):
        n_future_points = 4
        dp = DataPoint(self.dataset, future_points=n_future_points)
        col_name = "lowest_ask"

        actual = dp.get_future_values(col_name)
        cursor = dp.get_current_ts() + dp.period

        expected = self.dataset.loc[cursor:, col_name]

        for i in range(len(expected)):
            self.assertEqual(expected.iloc[i], actual.iloc[i])
        self.assertEqual(len(expected), len(actual))

if __name__ == '__main__':
    unittest.main()
