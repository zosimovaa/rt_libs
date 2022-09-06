from .core_facade import CoreFacade

from ..observation_builder import ObservationBuilderV2TradesSimpleBalance
from ..observation_builder import ObservationBuilderV2TradesBuySellFeats
from ..observation_builder import ObservationBuilderV2TradesRelativeBalance


class CoreV2TradesSimpleBalance(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2TradesSimpleBalance(self.context)


class CoreV2TradesBuySellFeats(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2TradesBuySellFeats(self.context)


class CoreV2TradesRelativeBalance(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderV2TradesRelativeBalance(self.context)
