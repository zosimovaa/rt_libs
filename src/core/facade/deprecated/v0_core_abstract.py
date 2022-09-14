from core.facade.core_facade import RTCore

# Task #1 - Sequence prediction
from core.context import AbstractContextSequencePrediction
from core.observation_builder import AbstractObservationBuilderSequencePrediction
from core.tickers import AbstractTickerBasic

# Task #2 - Close Signal
from core.context import AbstractContextCloseSignal
from core.observation_builder import AbstractObservationBuilderCloseSignal

# Task #3 - Open Signal
from core.context import AbstractContextOpenSignal
from core.tickers import AbstractTickerOpenSignal
from core.observation_builder import AbstractObservationBuilderOpenSignal

# Task #2 - Complete trade
from core.context import AbstractContextCompleteTrade
from core.tickers import AbstractTickerCompleteTrade
from core.observation_builder import AbstractObservationBuilderCompleteTrade


class CoreV0AbstractSequence(RTCore):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        RTCore.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextSequencePrediction()
        self.action_controller = AbstractTickerBasic(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderSequencePrediction(self.context)


class CoreV0AbstractCloseSignal(RTCore):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        RTCore.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextCloseSignal()
        self.action_controller = AbstractTickerBasic(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderCloseSignal(self.context)


class CoreV0AbstractOpenSignal(RTCore):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        RTCore.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextOpenSignal()
        self.action_controller = AbstractTickerOpenSignal(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderOpenSignal(self.context)


class CoreV0AbstractCompleteTrade(RTCore):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.0015):
        RTCore.__init__(self, penalty=penalty, reward=reward, market_fee=market_fee)
        self.context = AbstractContextCompleteTrade()
        self.action_controller = AbstractTickerCompleteTrade(self.context, penalty=penalty, reward=reward)
        self.observation = AbstractObservationBuilderCompleteTrade(self.context)
