import time
import uuid
import pickle
import tensorflow as tf

from .providers.fs import FsDataProvider
from .providers.db import DbDataProvider


class ScorePersistManager:
    """Сохраняет результаты локально и в БД"""

    TARGET_DIR = "scores"  # Директория для хранения удачных моделей
    NAME_PREFIX = "score"

    PUT_SCORE = "INSERT INTO scores VALUES"
    GET_SCORE = "SELECT * from scores !WHERE! ORDER BY frame ASC"

    PUT_WEIGHTS = "INSERT INTO weights VALUES"
    GET_WEIGHTS = "SELECT * from weights !WHERE! ORDER BY frame ASC"

    def __init__(self, alias, path, db_conf):
        self.alias = alias
        self.db_conf = db_conf
        self.fs = FsDataProvider(alias, path, self.TARGET_DIR, self.NAME_PREFIX)
        self.db_score = DbDataProvider(db_conf, self.GET_SCORE, self.PUT_SCORE)
        self.db_weights = DbDataProvider(db_conf, self.GET_WEIGHTS, self.PUT_WEIGHTS)

    def save(self, dataset, frame, metrics, weights):
        if self.db_conf is None:
            self.fs.put(frame, weights)
        else:
            weights_id = str(uuid.uuid4())

            weights_data = self._build_weights(weights_id, frame, weights)
            self.db_weights.put(weights_data)

            score_data = self._build_score(dataset, frame, metrics, weights_id)
            self.db_score.put(score_data)

    def read(self, frame):
        if self.db_conf is None:
            weights = self.fs.get(frame)
        else:
            condition = {"alias": self.alias, "frame": frame}
            weights = self.db_weights.get(**condition)

            weights = pickle.loads(weights[-1][-1])
        return weights

    def _build_weights(self, weights_id, frame, weights):
        data = [(
            weights_id,
            self.alias,
            frame,
            pickle.dumps(weights, protocol=pickle.HIGHEST_PROTOCOL)
        )]
        return data

    def _build_score(self, dataset, frame, metrics, weights_id):
        data = [(
            self.alias,
            dataset,
            frame,
            metrics,
            weights_id,
            int(time.time()),
            ""
        )]
        return data
