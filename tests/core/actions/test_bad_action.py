"""Модуль реализует тестирование класса BadAction"""

import unittest

from tests.test_dataset import TestDatasetGenerator

from src.data_point import DataPoint
from src.core.context import BasicContext
from src.core.actions import BadAction


class BadActionTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = TestDatasetGenerator().make(num=4, lowest_ask=100, highest_bid=99, ts_start=1000, price_step=1, period=60)
        data_point = DataPoint(self.dataset,  n_observation_points=4, n_future_points=0, period=1)
        self.context = BasicContext()
        self.context.update_datapoint(data_point)

    def test_bad_action(self):
        bad_action = BadAction(self.context)
        self.assertEqual(bad_action.ts, self.dataset.index.values[-1])
        self.assertEqual(bad_action.action, 0)


if __name__ == '__main__':
    unittest.main()
