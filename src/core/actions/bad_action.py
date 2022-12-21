import logging
from .base_action import BaseAction


class BadAction(BaseAction):
    """The class describes the wrong action of the neural network in the current context"""
    def __init__(self, context):
        BaseAction.__init__(self)
        self.ts = context.get("ts")
        self.action = context.get("action")
        self.is_open = context.get("is_open", domain="Trade")
