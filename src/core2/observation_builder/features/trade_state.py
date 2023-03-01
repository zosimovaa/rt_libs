"""
Module with basic features_del - deal status
"""

import logging

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class TradeStateFeature(BaseFeature):
    """Retrieves the current state of a trade operation from the context"""
    def __init__(self, context):
        super().__init__(context)

    def _get(self):
        trade_state = self.context.get("is_open", domain="Trade")
        #logger.debug("Get trade state() -> {}".format(trade_state))
        return trade_state
