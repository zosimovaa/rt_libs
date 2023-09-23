from .abstract_action import AbstractAction


class BadAction(AbstractAction):
    """The class describes the wrong action applied by a neural network."""
    def __init__(self, ts, is_open, info=None):
        AbstractAction.__init__(self, ts, is_open, info=info)
