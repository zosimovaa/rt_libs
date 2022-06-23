from .core_facade import CoreFacade

# Task #5 - Simple
from ..context import BasicContext
from ..observation_builder import ObservationBuilderBasic
from ..observation_builder import ObservationBuilderFutureFeatureCache, ObservationBuilderTrendIndicator
from ..tickers import TickerBasic, TickerExtendedReward


class TrainCoreSyntheticSimple(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        CoreFacade.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = BasicContext()
        self.action_controller = TickerBasic(self.context, penalty=penalty, reward=reward)
        self.observation = ObservationBuilderBasic(self.context)


class TrainCoreSyntheticExtendedReward(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        CoreFacade.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = BasicContext()
        self.action_controller = TickerExtendedReward(self.context, penalty=penalty, reward=reward)
        self.observation = ObservationBuilderBasic(self.context)


class TrainCoreSyntheticTrendIndicator(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        CoreFacade.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = BasicContext()
        self.action_controller = TickerExtendedReward(self.context, penalty=penalty, reward=reward)
        self.observation = ObservationBuilderTrendIndicator(self.context)