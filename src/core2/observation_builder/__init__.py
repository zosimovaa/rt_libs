"""
Classes which implements obserbaion building
"""
# Реализации классов билдера для первой части задачи - для абстрактных операций.
from .v0_abstract_builder import AbstractObservationBuilderSequencePrediction
from .v0_abstract_builder import AbstractObservationBuilderCloseSignal
from .v0_abstract_builder import AbstractObservationBuilderOpenSignal
from .v0_abstract_builder import AbstractObservationBuilderCompleteTrade


from .builder_1inp import ObservationBuilder1Inp
from .builder_2inp import ObservationBuilder2Inp

from .builder_inputs import ObservationBuilderInput

