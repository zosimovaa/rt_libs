# Реализации классов билдера для первой части задачи - для абстрактных операций.
from .v0_abstract_builder import AbstractObservationBuilderSequencePrediction
from .v0_abstract_builder import AbstractObservationBuilderCloseSignal
from .v0_abstract_builder import AbstractObservationBuilderOpenSignal
from .v0_abstract_builder import AbstractObservationBuilderCompleteTrade

from .v1_basic import ObservationBuilderBasic
from .v1_with_trend_indicator import ObservationBuilderTrendIndicator

from .v2_trades import ObservationBuilderV2ObTb
from .v2_trades import ObservationBuilderV2TradeBalance
from .v2_trades import ObservationBuilderV2Orderbook, ObservationBuilderV2OrderbookV2

from .v2_trades import ObservationBuilderV2OrderbookDiffFeature
