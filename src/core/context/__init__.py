# Базовый класс контекста
from .basic_context import BasicContext

# Реализации контекста для первой части задачи - для абстрактных операций
from .abstract_context import AbstractContextSequencePrediction
from .abstract_context import AbstractContextCloseSignal
from .abstract_context import AbstractContextOpenSignal
from .abstract_context import AbstractContextCompleteTrade
