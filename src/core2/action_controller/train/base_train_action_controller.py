"""
Базовый класс с роутингом для для action controller
"""
import logging

from ..base_action_router import BaseActionRouter

logger = logging.getLogger(__name__)


class BaseTrainActionController(BaseActionRouter):
    """
    Базовый класс с роутигном для реализации action controller. Необходимо переорпределить реализацию методов роутера
    """

    def __init__(self,
                 context=None,
                 penalty=-2, reward=0, market_fee=0.00155,
                 scale_wait=1, scale_open=1, scale_hold=1, scale_close=1):


        self.market_fee = market_fee

        self.scale_wait = scale_wait
        self.scale_open = scale_open
        self.scale_hold = scale_hold
        self.scale_close = scale_close

        super().__init__(context=context, penalty=penalty, reward=reward)

        self.context.set("market_fee", self.market_fee)
        self.reset()
        print("BaseTrainActionController")

