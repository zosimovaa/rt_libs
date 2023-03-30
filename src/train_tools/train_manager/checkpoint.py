
import pickle
import os
from tensorflow.python.keras.models import save_model, load_model


class SnapshotLord:
    SNAPSHOT_PATH = './train_snapshots/'
    MODELS_DIR = "models"
    SNAPSHOTS_DIR = "snapshots"

    def __init__(self, alias):
        self.path = os.path.join(self.SNAPSHOT_PATH, alias)
        self.models_path = os.path.join(self.path, self.MODELS_DIR)
        self.snapshots_path = os.path.join(self.path, self.SNAPSHOTS_DIR)
        self._check_dir(self.path)
        self._check_dir(self.models_path)
        self._check_dir(self.snapshots_path)

    @staticmethod
    def _check_dir(path):
        """Проверяет наличие директории и при необходимости ее создает"""
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model(self, model, model_name, path=None):
        """Сохраняет модель в процессе тренировки в папку"""
        if path is None:
            path = self.models_path
        full_path = os.path.join(path, str(model_name) + ".h5")
        save_model(model, full_path, overwrite=True, save_format="h5")

    def load_model(self, model_name, path=None):
        """Читает с диска сохраненную модель"""
        if path is None:
            path = self.models_path

        full_path = os.path.join(path, str(model_name) + ".h5")
        model = load_model(full_path, compile=True)
        return model

    def save_object(self, path, obj_name, obj):
        """Сохраняет произвольный объект"""
        obj_path = os.path.join(path, obj_name)

        with open(obj_path, 'wb') as stream:
            pickle.dump(obj, stream)

    def load_object(self, path, obj_name):
        """Загружает произвольный объект"""
        obj_path = os.path.join(path, obj_name)
        with open(obj_path, "rb") as stream:
            obj = pickle.load(stream)
        return obj