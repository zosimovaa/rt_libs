"""
This component implements the logic of the exchange
 - facade - implements the core for training and trading processes
 - action_handlers - describes the agent's current action and required context
 - context - single context of all components
 - observation_builder - builds an observation based on the current data point
 - action_controller - implements action reward logic

"""
from .actions import BadAction, TradeAction
from .facade import RTCore
from .constructor import ConstructorGen2
