import logging
from .base_action import BaseAction


logger = logging.getLogger(__name__)


class BadAction(BaseAction):
    """Класс описывает неверное действие нейросети в текущем контексте"""
    def __init__(self, ts, action):
        BaseAction.__init__(self)
        self.ts = ts
        self.action = action
        logger.debug("Bad action happened at {0} with id {1}".format(ts, self.id))
