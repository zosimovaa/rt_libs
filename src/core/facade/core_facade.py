"""
Торговая операция - передается при открытии и закрытии.
 - при обучении используется для сбора метрик
 - при тестировании использую в плеере. Такой подход позволит корректно показать открытую, но не закрытую операцию
 - при торговле - для фиксации в БД. Логикак как при сборе метрик

"""
import logging

from ..metrics import MetricCollector
from .core_error import RTCoreError

from basic_application import with_exception

logger = logging.getLogger(__name__)


class RTCore:
    COLLECT_METRICS = True
    """Реализация тренера с базовым набором фичей, без предсказания."""
    def __init__(self, context, action_controller, observation):
        self.context = context
        self.action_controller = action_controller
        self.observation_builder = observation

        self.metric_collector = MetricCollector()
        logger.debug("Instance initialized")

    @with_exception(RTCoreError)
    def get_action_space(self):
        # todo реализовать метод в action_controller
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
