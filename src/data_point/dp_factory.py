from .data_point import DataPoint
import numpy as np
import logging
from basic_application import with_exception

logger = logging.getLogger(__name__)


class DataPointFactoryError(Exception):
    pass


class DataPointFactory:
    def __init__(self, dataset, period=300, n_observation_points=10, n_history_points=0, n_future_points=0, step_size=None):
        self.period = period
        self.n_observation_points = n_observation_points
        self.n_history_points = n_history_points
        self.n_future_points = n_future_points
        self.n_points = n_observation_points + n_history_points + n_future_points
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = period

        self.dataset = dataset
        self.cursor = None
        self.max_step = None

        self.done = True

        self.reset()

    @with_exception(DataPointFactoryError)
    def reset(self, dataset=None):
        if dataset is not None:
            self.dataset = dataset

        self.max_step = max(self.dataset.index)
        self.done = False

        # Курсор устанавливается на значение, которое отстоит от начала так, чтобы от начала
        # до курсора был размер данных для одного датапоинта.
        self.cursor = min(self.dataset.index) + self.period * (self.n_points - 1)

        data_point = self.get_current_step()
        return data_point

    def get_idx(self):
        up_bound = min(self.cursor + self.period, self.dataset.index.max() + self.period)
        low_bound = up_bound - self.n_points * self.period
        idxs = np.arange(low_bound, up_bound, self.period)
        return idxs

    @with_exception(DataPointFactoryError)
    def get_current_step(self):
        idxs = self.get_idx()
        data = self.dataset.loc[idxs, :]

        data_point = DataPoint(
            data,
            n_observation_points=self.n_observation_points,
            n_future_points=self.n_future_points,
            period=self.period
        )
        return data_point

    @with_exception(DataPointFactoryError)
    def get_next_step(self):
        if not self.done:
            self.cursor = self.cursor + self.step_size

        if self.cursor >= self.max_step:
            self.done = True
        else:
            self.done = False

        data_point = self.get_current_step()
        return data_point, self.done
