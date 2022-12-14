import numpy as np
import pandas as pd


class TestDatasetGenerator:
    """Класс создает датасет для тестов"""
    @staticmethod
    def make(num=30, lowest_ask=96, highest_bid=95, ts_start=1000000, price_step=1, period=60):
        """Метод генерирует датасет по параметрам"""

        lowest_asks = np.arange(lowest_ask, lowest_ask + num, price_step).reshape(-1, 1)
        highest_bids = np.arange(highest_bid, highest_bid + num, price_step).reshape(-1, 1)
        ts = np.arange(ts_start, ts_start + num * period, period * price_step).reshape(-1, 1)

        dataset = np.concatenate([ts, lowest_asks, highest_bids], axis=1)
        dataset = pd.DataFrame(dataset, columns=["ts", "lowest_ask", "highest_bid"])
        dataset = dataset.set_index("ts")
        return dataset
