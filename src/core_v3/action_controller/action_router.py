"""
Модель содержит базовую реализацию обработчика событий без обработки действия.
По сути - только объявление действия и роутинг на нужный обработчик
"""
import logging

from ..context import ContextConsumer

logger = logging.getLogger(__name__)


class Router4Action(ContextConsumer):
    """
    Базовый абстрактный класс с роутигном для реализации обработки экшенов.
    """
    router = {
        0: "apply_action_wait",
        1: "apply_action_open",
        2: "apply_action_hold",
        3: "apply_action_close",
    }

    def __init__(self, alias):
        super().__init__(alias)

    def reset(self):
        """Reset current action controller state"""
        pass

    def apply_action(self, action):
        """The apply_action method applied action, calculating reward and open/close trade"""
        return getattr(self, self.router[action])()

    def apply_action_wait(self):
        raise NotImplementedError

    def apply_action_open(self):
        raise NotImplementedError

    def apply_action_hold(self):
        raise NotImplementedError

    def apply_action_close(self):
        raise NotImplementedError

    def get_action_space(self):
        return len(self.router)
