"""
Базовый класс дествия
"""

import uuid


class BaseAction:
    """Базовый класс действия"""
    def __init__(self):
        self.id = str(uuid.uuid4())
