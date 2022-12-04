"""
Trading operation - transmitted at the opening and closing.
  - in training is used to collect metrics
  - when testing, I use it in the player. This approach will correctly show an open but not closed operation.
  - when trading - for fixing in the database. Logic like when collecting metrics

"""
import logging

from ..metrics import MetricCollector
from .core_error import RTCoreError

from basic_application import with_exception

logger = logging.getLogger(__name__)


class RTCore:
    COLLECT_METRICS = True
    """Implementation of a trainer with a basic set of features, without prediction."""
    def __init__(self, context, action_controller, observation):
        self.context = context
        self.action_controller = action_controller
        self.observation_builder = observation

        self.metric_collector = MetricCollector()
        logger.debug("Instance initialized")

    @with_exception(RTCoreError)
    def get_action_space(self):
        # todo implement method in the action_controller
        action_space = len(self.action_controller.handler)
        logger.debug("Action space: {}".format(action_space))
        return action_space

    @with_exception(RTCoreError)
    def reset(self, data_point=None):
        logger.debug("Reset")
        self.context.reset()
        self.metric_collector.reset()
        self.context.update_datapoint(data_point)
        self.observation_builder.reset()
        self.action_controller.reset()

    @with_exception(RTCoreError)
    def get_observation(self, data_point):
        self.context.update_datapoint(data_point)
        observation = self.observation_builder.get(data_point)
        self.context.set("observation", observation, domain="Data")
        return observation

    @with_exception(RTCoreError)
    def apply_action(self, action):
        self.context.set("action", action, domain="Action")
        reward, action_result = self.action_controller.apply_action(action)
        self.context.set("reward", reward, domain="Action")
        if self.COLLECT_METRICS:
            self.metric_collector.process(reward, action_result)
        logger.debug("Action applied | reward: {0:.4f}".format(reward))
        logger.debug("Action type returned {0}".format(type(action_result)))
        return reward, action_result

    @with_exception(RTCoreError)
    def get_metrics(self):
        return self.metric_collector.get_metrics()
