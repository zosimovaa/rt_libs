from .core_facade import RTCore


class CoreV1Basic(RTCore):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        RTCore.__init__(self, *args, **kwargs)

