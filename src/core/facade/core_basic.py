from .core_facade import CoreFacade


class CoreBasic(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)

