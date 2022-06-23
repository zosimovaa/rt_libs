from .core_facade import CoreFacade
from ..observation_builder import ObservationBuilderTrendIndicator


class CoreTrendIndicator(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderTrendIndicator(self.context)
