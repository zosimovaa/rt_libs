"""
This component implements the logic of the exchange
 - facade - implements the core for training and trading processes
 - actions - describes the agent's current action and required context
 - context - single context of all components
 - observation_builder - builds an observation based on the current data point
 - tickers - implements action reward logic

"""
from .actions import BadAction, TradeAction

from .facade import RTCore

from core.constructor import ConstructorGen1

