import abc
import pandas as pd


class BasePrecompute(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process(self, data, *args, **kwargs) -> pd.DataFrame:
        pass
