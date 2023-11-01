import uuid


class AbstractAction:
    """Base action class"""
    def __init__(self, ts, is_open, info=None):
        self.ts = ts
        self.is_open = is_open
        self.info = info

        self.id = str(uuid.uuid4())