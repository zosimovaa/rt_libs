from .base_action import BaseAction


class BadAction(BaseAction):
    """Класс описывает неверное действие нейросети в текущем контексте"""
    def __init__(self, ts, action):
        BaseAction.__init__(self)
        self.ts = ts
        self.action = action
