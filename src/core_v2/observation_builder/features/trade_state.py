"""
Module with basic features_del - deal status
"""

import logging
import numpy as np

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class TradeStateSingleFeature(BaseFeature):
    """Retrieves the current state of a trade operation from the context"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get(self):
        feature = self.context.get("is_open")
        return np.array([feature])
