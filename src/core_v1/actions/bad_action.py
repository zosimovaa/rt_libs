from .base_action import BaseAction


class BadAction(BaseAction):
    """The class describes the wrong action of the neural network."""
    def __init__(self, ts, action, is_open):
        BaseAction.__init__(self)
        self.ts = ts
        self.action = action
        self.is_open = is_open
