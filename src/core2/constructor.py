from core.facade import RTCore

from core import context
from core import action_controller
from core import observation_builder
from core.observation_builder import features2


class ConstructorGen1:
    """Конструктор core из конфига"""

    def get_core(self, config):
        """Основной метод для сборки core"""
        context_config = config.get("context")
        context_instance = self._context_constructor(context_config)

        at_config = config.get("action_controller")
        action_controller_instance = self._ticker_constructor(at_config, context_instance)

        ob_config = config.get("observation_builder")
        observation_builder_instance = self._observation_builder_constructor(ob_config, context_instance)

        core = RTCore(context_instance, action_controller_instance, observation_builder_instance)
        return core

    def _context_constructor(self, config):
        """Метод собирает контекст"""
        class_name = config.get("class")
        params = config.get("params")
        context_instance = getattr(context, class_name)(**params)
        return context_instance

    def _ticker_constructor(self, config, context_instance):
        """Метод собирает экшн контроллер"""
        class_name = config.get("class")
        params = config.get("params")
        action_controller_instance = getattr(action_controller, class_name)(context_instance, **params)
        return action_controller_instance

    def _feature_builder(self, features_config, context_instance):
        """метод собирает конкретную фичу"""
        feature_list = []
        for feat_cfg in features_config:
            feature_class_name = feat_cfg.get("class")
            params = feat_cfg.get("params")

            if params is not None:
                feat_instance = getattr(features2, feature_class_name)(context_instance, **params)
            else:
                feat_instance = getattr(features2, feature_class_name)(context_instance)
            feature_list.append(feat_instance)

        return feature_list

    def _observation_builder_constructor(self, config, context_instance):
        """Метод собирает observation_builder из фичей"""
        class_name = config.get("class")

        static_config = config.get("features_del").get("static")
        series_config = config.get("features_del").get("series")

        static_feats = self._feature_builder(static_config, context_instance)
        series_feats = self._feature_builder(series_config, context_instance)

        observation_builder_class = getattr(observation_builder, class_name)
        observation_builder_instance = observation_builder_class(context_instance, static=static_feats, series=series_feats)
        return observation_builder_instance
