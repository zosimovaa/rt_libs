"""
В модуле описан интерфейс для реализации ObservationBuilder
"""
from abc import ABC, abstractmethod


class ObservationBuilderInterface(ABC):
    @abstractmethod
    def get(self, datapoint):
        """
        Метод формирует сэмл наблдения
        :param datapoint: DataPoint object
        :return: observation sample, np.array или list of np.array
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Метод сбрасывает состояние ObservationBuilder
        :return:
        """
        pass
