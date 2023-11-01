from .core import RTCore

from .action_controller import train_controllers


from .observation_builder import features
from .observation_builder import inputs
from .observation_builder import ObservationBuilder


class Constructor:
    """Конструктор core_v2 из конфига"""

    def get_core(self, alias, config):
        """Основной метод для сборки core_v1"""

        ac_config = config.get("action_controller")
        ac_instance = self._get_instance(alias, train_controllers, ac_config)

        ob_config = config.get("observation_builder")
        ob_instance = self._observation_builder_constructor(alias, ob_config)

        core = RTCore(alias, ac_instance, ob_instance)
        return core

    @staticmethod
    def _get_instance(alias, source, config):
        class_name = config.get("class")
        params = config.get("params", {})
        instance = getattr(source, class_name)(alias, **params)
        return instance

    def _observation_builder_constructor(self, alias, config):
        """Метод собирает observation_builder из фичей"""
        ob_class = config.get("class")
        inputs_config = config.get("inputs")

        inputs_instances = []
        for input_config in inputs_config:

            feature_instances = []
            features_config = input_config.get("features")
            for feature_config in features_config:
                feature_instance = self._get_instance(alias, features, feature_config)
                feature_instances.append(feature_instance)

            input_class = input_config.get("class")
            input_inst = getattr(inputs, input_class)(*feature_instances)

            inputs_instances.append(input_inst)

        ob_instance = ObservationBuilder(alias, inputs=inputs_instances)
        return ob_instance