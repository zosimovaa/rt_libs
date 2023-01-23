"""
Интерфейс для action controller
"""
from core.actions import BaseAction


class ActionControllerInterface():
    """
    Action controller interface
    """
    def reset(self):
        """Just reset current action controller state"""
        pass

    def apply_action(self, action) -> (int, BaseAction):
        """The apply_action method applied action and calculate reward"""
        pass

