import os
import yaml
import datetime


class ModelSaver:
    def __init__(self, models_path="./trader_models/", conf_name="config.yaml"):
        self.save_path = models_path
        self.conf_name = conf_name
        self.full_path = None

    def save(self, alias, model, tarder_config, observation_config, core_config, precompute_config):
        dt = datetime.datetime.now()

        self.full_path = os.path.join(self.save_path, alias + "|" + dt.strftime('%Y-%m-%d'))

        self.check_dir(self.full_path)
        self._save_model(model)
        self._save_config(tarder_config, observation_config, core_config, precompute_config)

    def _save_model(self, model):
        model.save(os.path.join(self.full_path, "model"))

    def _save_config(self, tarder_config, observation_config, core_config, precompute_config):
        config = {
            "trader": tarder_config,
            "observation": observation_config,
            "core": core_config,
            "precompute": precompute_config
        }
        with open(os.path.join(self.full_path, self.conf_name), mode='w', encoding="utf-8") as yml:
            yaml.dump(config, yml, allow_unicode=True)

    @staticmethod
    def check_dir(full_path):
        if not os.path.exists(full_path):
            os.makedirs(full_path)
