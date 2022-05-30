from .context import ContextWithDomains
from .ticker import Ticker
from .market_providers import TestMarketProvider
from .observation_builder.observation import ObservationBuilderBasicCache, ObservationBuilderBasic
from .observation_builder.observation import ObservationBuilderFutureFeatureCache, ObservationBuilderFutureFeature
from .metrics import MetricCollector
import logging
from abc import ABC, abstractmethod
from basic_application import with_exception

logger = logging.getLogger(__name__)


class CoreError(Exception):
    pass


class CoreFacadeInterface(ABC):
    """Интерфейс, описывающий функциональность для тренировки моделей и обеспечению реальных торгов на бирже"""

    @abstractmethod
    def get_action_space(self):
        """Получеие размерности action"""
        pass

    @abstractmethod
    def reset(self, data_point=None):
        """Сброс core в исходное состояние"""
        pass

    @abstractmethod
    def get_observation(self, data_point=None):
        """ Получение sample для текущей точки данных"""
        pass

    @abstractmethod
    def apply_action(self, action):
        """Применение действия и расчет награды и профита"""
        pass

    @abstractmethod
    def get_metrics(self):
        """Получение метрик процесса обучения"""
        pass


class CoreFacade:
    """Реализация тренера с базовым набором фичей, без предсказания."""
    def __init__(self, penalty=-2, reward=0, market_fee=0.00155, alias="test run", save_metrics=True):
        self.alias = alias
        self.market_fee = market_fee
        self.penalty = penalty
        self.reward = reward
        self.save_metrics = save_metrics

        self.log = logging.getLogger("{0}.{1}.{2}".format(__name__, self.__class__.__name__, self.alias))

        self.context = ContextWithDomains(market_fee=self.market_fee)
        self.trade_controller = TestMarketProvider(self.context, market_fee=self.market_fee)
        self.action_controller = Ticker(self.context, self.trade_controller)
        self.observation = ObservationBuilderBasicCache(self.context)
        self.metric_collector = MetricCollector()

        self.log.debug("Instance initialized")

    def get_action_space(self):
        # todo реализовать метод в action_controller
        action_space = len(self.action_controller.handler)
        self.log.debug("Action space: {}".format(action_space))
        return action_space

    def reset(self, data_point=None):
        self.log.debug("Reset")
        self.context.reset()
        self.context.update_datapoint(data_point)
        self.observation.reset()
        self.action_controller.reset()
        self.metric_collector.reset()

    @with_exception(CoreError)
    def get_observation(self, data_point=None):
        self.context.update_datapoint(data_point)
        # todo внедрить статус в процессы core
        self.context.set("status", "ok")
        observation = self.observation.get()
        self.context.set("observation_builder", observation, domain="Data")
        return observation

    @with_exception(CoreError)
    def apply_action(self, action):
        self.context.set("action", action)
        reward, action_result = self.action_controller.apply_action(action)
        if self.save_metrics:
            self.metric_collector.process(reward, action_result)
        self.log.debug("Action applied | reward: {0:.4f}".format(reward))
        self.log.debug("Action type returned {0}".format(type(action_result)))
        return reward, action_result

    def get_metrics(self):
        return self.metric_collector.get_metrics()


class TrainCoreBasic(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)


class TrainCoreFutureFeature(CoreFacade):
    """Реализация тренера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderFutureFeatureCache(self.context)


class TradeCoreBasic(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderBasic(self.context)
        self.save_metrics = False


class TradeCoreFutureFeature(CoreFacade):
    """Реализация трейдера с базовым набором фичей, с предсказанием."""
    def __init__(self, *args, **kwargs):
        CoreFacade.__init__(self, *args, **kwargs)
        self.observation = ObservationBuilderFutureFeature(self.context)
        self.save_metrics = False

