import uuid


class BaseAction:
    """Base action class"""
    def __init__(self):
        self.id = str(uuid.uuid4())
