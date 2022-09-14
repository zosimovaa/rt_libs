from .core_facade import RTCore
from ..observation_builder import ObservationBuilderTrendIndicator


class CoreV1TrendIndicator(RTCore):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        RTCore.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderTrendIndicator(self.context)
