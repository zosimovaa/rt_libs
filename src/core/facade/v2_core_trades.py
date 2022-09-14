from .core_facade import CoreFacade

from ..observation_builder import ObservationBuilderV2TradeBalance
from ..observation_builder import ObservationBuilderV2Orderbook, ObservationBuilderV2OrderbookV2
from ..observation_builder import ObservationBuilderV2ObTb
from ..observation_builder import ObservationBuilderV2OrderbookDiffFeature


class CoreV2TradeBalance(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2TradeBalance(self.context)


class CoreV2Orderbook(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2Orderbook(self.context)

class CoreV2OrderbookV2(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2OrderbookV2(self.context)


class CoreV2ObTb(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2ObTb(self.context)


class CoreV2ObDiffFeat(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2OrderbookDiffFeature(self.context)
