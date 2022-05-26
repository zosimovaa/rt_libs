from .data_point import DataPoint
import numpy as np
import logging


class DataPointFactoryError(Exception):
    pass


class DataExpiredError(DataPointFactoryError):
    pass


class DataShapeError(DataPointFactoryError):
    pass


class DataPointFactoryInterface:
    def get_data_point(self, *args, **kwargs):
        pass


class DataPointFactoryBasic(DataPointFactoryInterface):
    def __init__(self, ts, data, n_observation_points, n_future_points, update_period):
        self.ts = ts
        self.data = data
        self.log = logging.getLogger("{0}.{1}".format(__name__, self.__class__.__name__))

        self.n_observation_points = n_observation_points
        self.n_future_points = n_future_points
        self.n_total_points = n_observation_points + n_future_points
        self.update_period = update_period

    def get_data_point(self, pair):
        dataset = self._get_dataset(pair)
        self._check_ts(dataset)
        self._check_shape(dataset)
        data_point = DataPoint(dataset, self.n_observation_points, self.n_future_points)
        self.log.debug("Datapoint returned with len: {0}".format(len(dataset)))
        return data_point

    def _get_dataset(self, pair):
        try:
            dataset = self.data.loc[self.data["pair"] == pair, ["lowest_ask", "highest_bid"]]
            dataset = dataset.replace(to_replace=0, method='ffill')
        except Exception as e:
            raise DataPointFactoryError from e
        return dataset

    def _check_ts(self, dataset):
        dataset_max_ts = max(dataset.index)
        if np.abs(self.ts - dataset_max_ts) > self.update_period:
            self.log.debug("dataset_max_ts {}".format(dataset_max_ts))
            self.log.debug("self.ts {}".format(self.ts))
            raise DataExpiredError

    def _check_shape(self, dataset):
        if len(dataset) != self.n_total_points:
            # todo здесь должна быть ошибка и логирование
            self.log.debug("Dataset shape error: {0}".format(dataset))
            raise DataShapeError


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
        up_bound = self.cursor + self.period
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

