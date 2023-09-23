import logging
import numpy as np

from .base_input import BaseInput
logger = logging.getLogger(__name__)


class Input2D(BaseInput):
    def __init__(self, *features):
        super().__init__(*features)

    def get(self):
        data = [feat.get() for feat in self.features]
        data = np.concatenate(data, axis=1)
        return np.array(data, dtype=np.float32)
