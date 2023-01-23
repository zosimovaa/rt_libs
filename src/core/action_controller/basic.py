import logging
import numpy as np

from .action_controller_interface import ActionControllerInterface
from ..actions import BadAction, TradeAction, OppositeTradeAction
from ..observation_builder.features import ProfitFeature

logger = logging.getLogger(__name__)


class ActionControllerBasic(ActionControllerInterface):
    """Класс реализует логику расчета награды/штрафа за действия.
    Базовая версия - награда из профита выдается только при закрытии.
    В WAIT, OPEN, HOLD - будет награда только в виде штрафа за неправильные действия.
    """
    handler = {
        0: "_action_wait",
        1: "_action_open",
        2: "_action_hold",
        3: "_action_close"
    }

    def __init__(self,
                 context,
                 penalty=-2, reward=0,
                 scale_wait=10, scale_open= 10, scale_hold= 10, scale_close=100,
                 num_mean_obs=2
                 ):
        self.context = context
        self.penalty = penalty
        self.reward = reward
        self.scale_wait = scale_wait
        self.scale_open = scale_open
        self.scale_hold = scale_hold
        self.scale_close = scale_close
        self.num_mean_obs = num_mean_obs

        self.trade = None
        logger.info("Initialized with penalty {0} and reward {1}.".format(penalty, reward))

    def reset(self):
        self.trade = None

    def apply_action(self, action):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open", domain="Trade")
        handler = getattr(self, self.handler[action])
        reward, action_result = handler(ts, is_open)
        return reward, action_result

    def _action_wait(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            reward = self.reward * self.scale_wait
            action_result = None
        return reward, action_result

    def _action_open(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            self.trade = TradeAction(self.context)
            self.context.set_trade(self.trade)
            action_result = self.trade

            reward = self.reward * self.scale_open
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            reward = self.reward * self.scale_hold
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _action_close(self, ts, is_open):
        if is_open:
            profit = self.context.get("profit", domain="Trade")
            reward = profit * self.scale_close
            self.trade.close()

            action_result = self.trade
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _get_penalty(self, val=None):
        """Расчет штрафа. Если штрафне задан явно, то берем из базового значения"""
        value = self.penalty if val is None else val
        logger.debug("_get_penalty(): -> {0}".format(value))
        return value



class ActionControllerBasicOpposite(ActionControllerBasic):

    def __init__(self, *args, **kwargs):
        ActionControllerBasic.__init__(self, *args, **kwargs)

    def reset(self):
        # Инициализация торговой операции для работы с просадкой
        self.trade = None
        opposite_trade = OppositeTradeAction(self.context)
        self.context.set("trade", opposite_trade, domain="OppositeTrade")

    def _action_open_trade(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            # Открыть сделку и изменить статус в контексте
            action_result = TradeAction(self.context)
            self.context.set_trade(action_result)

            # Закрыть opposite_trade и рассчитать награду
            opposite_trade = self.context.get("trade", domain="OppositeTrade")
            opposite_trade.close()
            #РАЗОБРАТЬСЯ
            reward = -opposite_trade.profit * self.scale_open

        return reward, action_result

    def _action_close_trade(self, ts, is_open):
        if is_open:
            highest_bid = self.context.get("highest_bid")
            # Закрыть сделку
            action_result = self.context.trade
            action_result.close()
            reward = action_result.profit * self.reward_close

            # Открыть opposite_trade
            self.opposite_trade = OppositeTradeAction(self.context)
            self.context.set("trade", self.opposite_trade, domain="OppositeTrade")

        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result



