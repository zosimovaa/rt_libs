from .data_point import DataPoint
import numpy as np
import logging


class DataPointFactory:
    def __init__(self, dataset=None, period=300, n_observation_points=5, n_future_points=3, step_size=60):
        self.period = period
        self.n_observation_points = n_observation_points
        self.n_future_points = n_future_points
        self.step_size = step_size

        self.dataset = dataset
        self.cursor = None
        self.max_step = None

        self.done = True

        if dataset is not None:
            self.reset()

    def reset(self, dataset=None):
        if dataset is not None:
            self.dataset = dataset

        self.max_step = max(self.dataset.index)
        self.done = False
        self.cursor = min(self.dataset.index) + self.period * (self.n_observation_points - 1)

        data_point = self.get_current_step()
        return data_point

    def get_idx(self):
        up_bound = min(self.cursor + self.period, self.dataset.index.max())
        low_bound = up_bound - self.n_observation_points * self.period
        idxs = np.arange(low_bound, up_bound, self.period)
        return idxs

    def get_future_idx(self):
        low_bound = self.cursor + self.period
        up_bound = self.cursor + (self.n_future_points + 1) * self.period
        idxs = np.arange(low_bound, up_bound, self.period)
        idxs = np.array(list(map(lambda x: min(x, self.max_step), idxs)))
        return (idxs)

    def get_current_step(self):
        idxs = self.get_idx()
        data_ = self.dataset.loc[idxs, ["lowest_ask", "highest_bid"]]

        idxs = self.get_future_idx()
        data_f = self.dataset.loc[idxs, ["lowest_ask", "highest_bid"]]

        data_point = DataPoint(data_, dataset_future=data_f)
        return data_point

    def get_next_step(self):
        if not self.done:
            self.cursor += self.step_size
        self.done = True if self.cursor >= self.max_step else False
        data_point = self.get_current_step()
        return data_point, self.done

