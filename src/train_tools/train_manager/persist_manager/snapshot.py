import time
import uuid
import pickle
import tensorflow as tf

from .providers.fs import FsDataProvider
from .providers.db import DbDataProvider


class ScorePersistManager(BasicFileManager):
    """Сохраняет результаты локально и в БД"""

    TARGET_DIR = "scores"  # Директория для хранения удачных моделей
    NAME_PREFIX = "score"

    PUT_SCORE = "INSERT INTO scores VALUES"
    GET_SCORE = "SELECT * from scores !WHERE! ORDER BY frame ASC"

    PUT_WEIGHTS = "INSERT INTO weights VALUES"
    GET_WEIGHTS = "SELECT * from weights !WHERE! ORDER BY frame ASC"

    def __init__(self, alias, path, db_conf):
        super().__init__(alias, path)

        self.db_conf = db_conf
        self.fs = FsDataProvider()

        self.db_score = DbDataProvider(db_conf, self.GET_SCORE, self.PUT_SCORE)
        self.db_weights = DbDataProvider(db_conf, self.GET_WEIGHTS, self.PUT_WEIGHTS)

    def save(self, dataset, frame, metrics, weights):
        if self.db_conf in None:
            path = self.get_path(self.TARGET_DIR)
            name = self.NAME_PREFIX + str(frame)
            self.fs.put(path, name, weights)
        else:
            weights_id = uuid.uuid4()

            weights_data = self._build_weights(weights_id, frame, weights)
            self.db_weights.put(weights_data)

            score_data = self._build_score(weights_id, frame, weights)
            self.db_score.put(score_data)

    def get_weights(self, frame):
        if self.db_conf in None:
            path = self.get_path(self.TARGET_DIR)
            name = self.NAME_PREFIX + str(frame)
            weights = self.fs.get(path, name)
        else:
            pass

        return weights

    def _build_weights(self, weights_id, frame, weights):
        return [(
            weights_id,
            self.alias,
            frame,
            pickle.dumps(weights, protocol=pickle.HIGHEST_PROTOCOL)
        )]

    def _build_score(self, dataset, frame, metrics, weights_id):
        return [(
            self.alias,
            dataset,
            frame,
            metrics,
            weights_id,
            int(time.time()),
            ""
        )]


