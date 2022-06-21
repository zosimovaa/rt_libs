from .core_facade import CoreFacade

# Task #1 - Sequence prediction
from ..context import AbstractContextSequencePrediction
from ..observation_builder import AbstractObservationBuilderSequencePrediction
from ..tickers import AbstractTickerBasic

# Task #2 - Close Signal
from ..context import AbstractContextCloseSignal
from ..observation_builder import AbstractObservationBuilderCloseSignal

# Task #3 - Open Signal
from ..context import AbstractContextOpenSignal
from ..tickers import AbstractTickerOpenSignal
from ..observation_builder import AbstractObservationBuilderOpenSignal

# Task #2 - Complete trade
from ..context import AbstractContextCompleteTrade
from ..tickers import AbstractTickerCompleteTrade
from ..observation_builder import AbstractObservationBuilderCompleteTrade


class TrainCoreAbstractSequence(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        CoreFacade.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextSequencePrediction()
        self.action_controller = AbstractTickerBasic(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderSequencePrediction(self.context)


class TrainCoreAbstractCloseSignal(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        CoreFacade.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextCloseSignal()
        self.action_controller = AbstractTickerBasic(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderCloseSignal(self.context)


class TrainCoreAbstractOpenSignal(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        CoreFacade.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextOpenSignal()
        self.action_controller = AbstractTickerOpenSignal(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderOpenSignal(self.context)


class TrainCoreAbstractCompleteTrade(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        CoreFacade.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextCompleteTrade()
        self.action_controller = AbstractTickerCompleteTrade(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderCompleteTrade(self.context)
