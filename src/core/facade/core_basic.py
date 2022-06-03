from .core_facade import CoreFacade
from ..observation_builder import ObservationBuilderBasic


class TrainCoreBasic(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)


class TradeCoreBasic(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderBasic(self.context)
