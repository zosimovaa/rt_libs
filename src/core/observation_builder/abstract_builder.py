"""
Модуль реализует набор абстрактных билдеров для предварительного обучения и настройки алгоритма
"""

import logging
import numpy as np

from .interface import ObservationBuilderInterface

logger = logging.getLogger(__name__)


class AbstractObservationBuilderSequencePrediction(ObservationBuilderInterface):
    """Билдер для обучения корректной последовательности (Task #1)"""
    def __init__(self, context):
        self.context = context

    def reset(self):
        pass

    def get(self, data_point):
        # trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # observation
        observation = np.array([trade_state], dtype=np.float32)
        return observation


class AbstractObservationBuilderCloseSignal(ObservationBuilderInterface):
    """Билдер для обучения корректной последовательности (Task #2)"""
    def __init__(self, context):
        self.context = context

    def reset(self):
        pass

    def get(self, data_point):
        # trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # close signal value
        close_signal = self.context.data_point.get_value("close_signal")

        # observation
        observation = np.array([trade_state, close_signal], dtype=np.float32)
        return observation


class AbstractObservationBuilderOpenSignal(ObservationBuilderInterface):
    """Билдер для обучения корректной последовательности (Task #3)"""
    def __init__(self, context):
        self.context = context

    def reset(self):
        pass

    def get(self, data_point):
        # trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # current signal value
        open_signal = self.context.data_point.get_value("open_signal")

        # open trade feat value
        if self.context.trade is not None and self.context.trade.is_open:
            open_signal_value = self.context.get("open_signal_value")
        else:
            open_signal_value = 0

        # observation
        observation = np.array([trade_state, open_signal, open_signal_value], dtype=np.float32)
        return observation


class AbstractObservationBuilderCompleteTrade(ObservationBuilderInterface):
    """Билдер для обучения корректной последовательности (Task #4)"""

    def __init__(self, context):
        self.context = context

    def reset(self):
        pass

    def get(self, data_point):
        # trade state feature
        trade_state = self.context.get("is_open", domain="Trade")

        # data
        open_signal = self.context.data_point.get_value("open_signal")
        close_signal = self.context.data_point.get_value("close_signal")

        # open trade feat value
        if self.context.trade is not None and self.context.trade.is_open:
            open_signal_value = self.context.get("open_signal_value")
            if open_signal_value < 0:
                open_signal_value = 0
            else:
                open_signal_value = 1
        else:
            open_signal_value = 0

        # observation
        observation = np.array([trade_state, open_signal, close_signal, open_signal_value], dtype=np.float32)
        return observation



