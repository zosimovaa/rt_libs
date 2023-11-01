"""
Оптимизированная структура экшенов
- AbstractAction
  - BadAction
  - CorrectAction
    - TradeAction
    - VoidAction
    - FailAction

"""

from .abstract_action import AbstractAction
from .bad_action import BadAction
from .correct_action import CorrectAction
from .correct_action import VoidAction, TradeAction, FailAction
