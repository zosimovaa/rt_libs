'''from .core_facade import RTCore

from ..observation_builder import ObservationBuilderTradeBalance
from ..observation_builder import ObservationBuilderV2Orderbook, ObservationBuilderV2OrderbookV2
from ..observation_builder import ObservationBuilderV2ObTb
from ..observation_builder import ObservationBuilderV2OrderbookDiffFeature


class CoreV2TradeBalance(RTCore):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        RTCore.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderTradeBalance(self.context)


class CoreV2Orderbook(RTCore):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        RTCore.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2Orderbook(self.context)

class CoreV2OrderbookV2(RTCore):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        RTCore.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2OrderbookV2(self.context)


class CoreV2ObTb(RTCore):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        RTCore.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2ObTb(self.context)


class CoreV2ObDiffFeat(RTCore):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        RTCore.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2OrderbookDiffFeature(self.context)
'''