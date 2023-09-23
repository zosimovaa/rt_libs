import logging

from ..context import ContextConsumer
logger = logging.getLogger(__name__)


class ObservationBuilder(ContextConsumer):
    def __init__(self, alias, inputs=()):
        super().__init__(alias=alias)
        self.inputs = inputs

    def reset(self):
        for inp in self.inputs:
            inp.reset()

    def get(self):
        if len(self.inputs) == 1:
            observation = self.inputs[0].get()
        else:
            observation = [inp.get() for inp in self.inputs]
        return observation
