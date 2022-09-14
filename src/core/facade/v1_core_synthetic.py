from .core_facade import RTCore

# Task #5 - Simple
from ..context import BasicContext
from ..observation_builder import ObservationBuilderBasic
from ..observation_builder import ObservationBuilderTrendIndicator
from ..tickers import TickerBasic, TickerExtendedReward


class CoreV1SyntheticSimple(RTCore):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        RTCore.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = BasicContext()
        self.action_controller = TickerBasic(self.context, penalty=penalty, reward=reward)
        self.observation = ObservationBuilderBasic(self.context)


class CoreV1SyntheticExtendedReward(RTCore):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        RTCore.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = BasicContext()
        self.action_controller = TickerExtendedReward(self.context, penalty=penalty, reward=reward)
        self.observation = ObservationBuilderBasic(self.context)


class CoreV1SyntheticTrendIndicator(RTCore):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        RTCore.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = BasicContext()
        self.action_controller = TickerExtendedReward(self.context, penalty=penalty, reward=reward)
        self.observation = ObservationBuilderTrendIndicator(self.context)