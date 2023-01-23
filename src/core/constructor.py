from core.facade import RTCore
from core import  context


from core import context
from core import tickers
from core import action_controller
from core import observation_builder
from core.observation_builder import features


class ConstructorGen1:
    """Конструктор core из конфига"""

    def get_core(self, config):
        """Основной метод для сборки core"""
        context_config = config.get("context")
        context = self._context_constructor(context_config)

        at_config = config.get("action_controller")
        action_controller = self._ticker_constructor(at_config, context)

        ob_config = config.get("observation_builder")
        observation_builder = self._observation_builder_constructor(ob_config, context)

        core = RTCore(context, action_controller, observation_builder)
        return core

    def _context_constructor(self, config):
        """Метод собирает контекст"""
        class_name = config.get("class")
        params = config.get("params")
        context_instance = getattr(context, class_name)(**params)
        return context_instance

    def _ticker_constructor(self, config, context):
        """Метод собирает экшн контроллер"""
        class_name = config.get("class")
        params = config.get("params")
        action_controller_instance = getattr(action_controller, class_name)(context, **params)
        return action_controller_instance

    def _feature_builder(self, features_config, context):
        """метод собирает конкретную фичу"""
        feature_list = []
        for feat_cfg in features_config:
            feature_class_name = feat_cfg.get("class")
            params = feat_cfg.get("params")

            if params is not None:
                feat_instance = getattr(features, feature_class_name)(context, **params)
            else:
                feat_instance = getattr(features, feature_class_name)(context)
            feature_list.append(feat_instance)

        return feature_list

    def _observation_builder_constructor(self, config, context):
        """Метод собирает observation_builder из фичей"""
        class_name = config.get("class")

        static_config = config.get("features").get("static")
        series_config = config.get("features").get("series")

        static_feats = self._feature_builder(static_config, context)
        series_feats = self._feature_builder(series_config, context)

        observation_builder_class = getattr(observation_builder, class_name)
        observation_builder_instance = observation_builder_class(context, static=static_feats, series=series_feats)
        return observation_builder_instance
