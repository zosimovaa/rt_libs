import os
import yaml


class ModelSaver:
    def __init__(self, alias, models_path="./trader_models/", conf_name="config.yaml"):
        self.alias = alias
        self.save_path = models_path
        self.conf_name = conf_name
        self.full_path = os.path.join(models_path, alias)

    def save(self, model, config):
        self.check_dir(self.full_path)
        self._save_model(model)
        self._save_config(config)

    def _save_model(self, model):
        model.save(os.path.join(self.full_path, "model"))

    def _save_config(self, config):
        with open(os.path.join(self.full_path, self.conf_name), mode='w', encoding="utf-8") as yml:
            yaml.dump(config, yml, allow_unicode=True)

    @staticmethod
    def check_dir(full_path):
        if not os.path.exists(full_path):
            os.makedirs(full_path)
