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
            open_feat_val = self.context.get("open_feat_val")
        else:
            open_feat_val = 0

        # observation
        observation = np.array([trade_state, open_signal, open_feat_val], dtype=np.float32)
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

        # open signal value
        open_signal = self.context.data_point.get_value("open_signal")

        # close signal value
        close_signal = self.context.data_point.get_value("close_signal")

        # open trade feat value
        if self.context.trade is not None and self.context.trade.is_open:
            open_feat_val = self.context.get("open_feat_val")
        else:
            open_feat_val = 0

        # observation
        observation = np.array([trade_state, open_signal, close_signal, open_feat_val], dtype=np.float32)
        return observation



