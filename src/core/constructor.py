from .facade import RTCore

from core import context
from core import action_controller
from core import observation_builder
from .observation_builder import features
from .observation_builder import inputs


class ConstructorGen2:
    """Конструктор core из конфига"""

    def get_core(self, config):
        """Основной метод для сборки core"""
        context_config = config.get("context")
        context_instance = self._get_instance(context, context_config)

        ac_config = config.get("action_controller")
        action_controller_instance = self._get_instance(action_controller, ac_config, context=context_instance)

        ob_config = config.get("observation_builder")
        observation_builder_instance = self._observation_builder_constructor(ob_config, context_instance)

        core = RTCore(context_instance, action_controller_instance, observation_builder_instance)
        return core

    def _get_instance(self, source, config, context=None):
        class_name = config.get("class")
        params = config.get("params")
        if context is not None:
            instance = getattr(source, class_name)(context, **params)
        else:
            instance = getattr(source, class_name)(**params)
        return instance

    def _observation_builder_constructor(self, config, context_instance):
        """Метод собирает observation_builder из фичей"""
        ob_class = config.get("class")
        inputs_config = config.get("inputs")

        inputs_instances = []
        for input_config in inputs_config:

            feature_instances = []
            features_config = input_config.get("features")
            for feature_config in features_config:
                feature_instance = self._get_instance(features, feature_config, context=context_instance)
                feature_instances.append(feature_instance)

            input_class = input_config.get("class")
            input_inst = getattr(inputs, input_class)(*feature_instances)

            inputs_instances.append(input_inst)

        ob_instance = getattr(observation_builder, ob_class)(inputs_instances)
        return ob_instance
