"""
Интерфейс для action controller
"""

import ABC
from ABC import abstracmethod

from core.actions import BaseAction


class ActionControllerInterface():
    """
    Action controller interface
    """
    @abstracmethod
    def reset(self):
        """Just reset current action controller state"""
        pass

    @abstracmethod
    def apply_action(self, action) -> (int, BaseAction):
        """The apply_action method applied action and calculate reward"""
        pass

