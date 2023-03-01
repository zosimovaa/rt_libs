import logging

from .base_builder import BaseObservationBuilder


logger = logging.getLogger(__name__)


class ObservationBuilderInput(BaseObservationBuilder):
    def __init__(self, *inputs):
        self.inputs = inputs

    def reset(self):
        for inp in self.inputs:
            inp.reset()

    def get(self):
        if len(self.inputs) == 1:
            return self.inputs[0].get()
        else:
            return [inp.get() for inp in self.inputs]
