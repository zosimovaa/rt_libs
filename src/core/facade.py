"""
Фасад
Выполняет оркестрацию запросов к observation_builder и action_controller
Занимается подготовкой контекста для потребителей.

"""
import logging

from .metrics import MetricCollector
from .errors import RTCoreError

from basic_application import with_exception

logger = logging.getLogger(__name__)


class RTCore:
    COLLECT_METRICS = True

    """Implementation of a trainer with a basic set of features."""
    def __init__(self, context, action_controller, observation_builder):
        self.context = context
        self.action_controller = action_controller
        self.observation_builder = observation_builder
        self.metric_collector = MetricCollector()
        logger.debug("Instance initialized")

    #@with_exception(RTCoreError)
    def get_action_space(self):
        # todo implement method in the action_controller
        action_space = len(self.action_controller.router)
        return action_space

    #@with_exception(RTCoreError)
    def reset(self, data_point=None):
        self.metric_collector.reset()
        self.context.reset()
        if data_point is not None:
            self.context.set_dp(data_point)
        self.observation_builder.reset()
        self.action_controller.reset()

    #@with_exception(RTCoreError)
    def get_observation(self, data_point):
        self.context.set_dp(data_point)
        observation = self.observation_builder.get()
        self.context.set("observation", observation)
        return observation

    #@with_exception(RTCoreError)
    def apply_action(self, action):
        reward, action_result = self.action_controller.apply_action(action)
        if self.COLLECT_METRICS:
            is_open = self.context.get("is_open")
            self.metric_collector.process(reward, action_result, is_open)
        return reward, action_result

    #@with_exception(RTCoreError)
    def get_metrics(self):
        return self.metric_collector.get_metrics()
