import numpy as np
import time
from .data_point import DataPoint


def with_debug_time(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print("{0:<15} | Exec time : {1:.8}".format(func.__name__, t1 - t0))
        return result

    return wrapper


class DatasetHandler:
    def __init__(self, dataset, offset, future_points=5):
        self.dataset = dataset
        self.n_steps = len(dataset)
        self.offset = offset
        self.cursor = offset
        self.future_points = future_points
        self.done = True
        self.data_point = None

    def reset(self):
        self.done = False
        self.cursor = self.offset
        self.data_point = self.get_current_step()
        return self.data_point

    def get_idx(self):
        bound_low = self.cursor - self.offset
        bound_hi = self.cursor + self.future_points
        idxs = np.arange(bound_low, bound_hi)
        idxs = np.array(list(map(lambda x: min(x, self.n_steps-1), idxs)))
        return idxs

    def get_dataset(self):
        idxs = self.get_idx()
        batch = self.dataset.iloc[idxs, :]
        return batch

    def get_current_step(self):
        dataset = self.get_dataset()
        data_point = DataPoint(dataset, self.offset, self.future_points)
        return data_point

    def get_next_step(self):
        if not self.done:
            self.cursor += 1
        self.done = True if self.cursor == self.n_steps else False
        self.data_point = self.get_current_step()
        return self.data_point, self.done
