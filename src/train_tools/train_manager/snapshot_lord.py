import pickle
import yaml
import os
from tensorflow.python.keras.models import save_model, load_model


class SnapshotLord:

    def __init__(self, paths):
        self.absolute_path = os.path.join(*paths)
        self._check_dir(self.absolute_path)

    @staticmethod
    def _check_dir(path):
        """Проверяет наличие директории и при необходимости ее создает"""
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model(self, paths, model_name, model, format="h5"):
        """Сохраняет модель в процессе тренировки в папку"""
        path = os.path.join(self.absolute_path, *paths)
        self._check_dir(path)
        if format=='h5':
            path = os.path.join(path, str(model_name) + ".h5")
        else:
            path = os.path.join(path, str(model_name))
        save_model(model, path, overwrite=True, save_format=format)

    def load_model(self, paths, model_name):
        """Читает с диска сохраненную модель"""
        path = os.path.join(self.absolute_path, *paths, str(model_name) + ".h5")
        model = load_model(path, compile=True)
        return model

    def save_object(self, paths, obj_name, obj):
        """Сохраняет произвольный объект"""
        path = os.path.join(self.absolute_path, *paths)
        self._check_dir(path)
        path = os.path.join(path, obj_name)
        with open(path, 'wb') as stream:
            pickle.dump(obj, stream)

    def load_object(self, paths, obj_name):
        """Загружает произвольный объект"""
        path = os.path.join(self.absolute_path, *paths, obj_name)
        with open(path, "rb") as stream:
            obj = pickle.load(stream)
        return obj

    def save_config(self, paths, name, config):
        path = os.path.join(self.absolute_path, *paths)
        self._check_dir(path)
        path = os.path.join(path, name)
        with open(path, mode='w', encoding="utf-8") as yml:
            yaml.dump(config, yml, allow_unicode=True)

    def load_config(self, paths, name):
        """Загружает конфиг"""
        path = os.path.join(self.absolute_path, *paths, name)
        with open(path, "r") as stream:
            config = yaml.safe_load(stream)
        return config
