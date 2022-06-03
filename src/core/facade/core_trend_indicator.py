from .core_facade import CoreFacade
from ..observation_builder import ObservationBuilderFutureFeatureCache
from ..observation_builder import ObservationBuilderFutureFeature


class TrainCoreFutureFeature(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderFutureFeatureCache(self.context)


class TradeCoreFutureFeature(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderFutureFeature(self.context)