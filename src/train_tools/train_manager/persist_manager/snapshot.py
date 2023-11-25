import time
import pickle

from .providers.fs import FsDataProvider
from .providers.db import DbDataProvider


class SnapshotPersistManager:
    """Сохраняет результаты локально и в БД"""

    TARGET_DIR = "snapshots"  # Директория для хранения
    NAME_PREFIX = "snapshot"

    PUT = "INSERT INTO snapshots VALUES"
    GET = "SELECT * from snapshots !WHERE!"

    def __init__(self, alias, path, db_conf):
        self.alias = alias
        self.db_conf = db_conf
        self.fs = FsDataProvider(alias, path, self.TARGET_DIR, self.NAME_PREFIX)
        self.db = DbDataProvider(db_conf, self.GET, self.PUT)

    def save(self, frame, snapshot):
        if self.db_conf is None:
            self.fs.put(frame, snapshot)
        else:
            data = self._build_data(frame, snapshot)
            self.db.put(data)

    def read(self, frame):
        if self.db_conf is None:
            snapshot = self.fs.get(frame)
        else:
            condition = {"alias": self.alias, "frame": frame}
            snapshot = self.db.get(**condition)
            snapshot = pickle.loads(snapshot[-1][2])
        return snapshot

    def _build_data(self, frame, snapshot):
        data = [(
            self.alias,
            frame,
            pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL),
            int(time.time())
        )]
        return data


