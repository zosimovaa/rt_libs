"""
Базовый класс с роутингом для для action controller
"""
import logging

logger = logging.getLogger(__name__)


class BaseActionRouter:
    """
    Базовый класс с роутигном для реализации action controller. Необходимо переорпределить реализацию методов роутера
    """
    router = {
        0: "apply_action_wait",
        1: "apply_action_open",
        2: "apply_action_hold",
        3: "apply_action_close",
    }

    def __init__(self, context=None, penalty=-2, reward=0):
        self.context = context
        self.penalty = penalty
        self.reward = reward
        self.trade = None
        #self.reset()

    def reset(self):
        """Reset current action controller state"""
        self.trade = None
        self.context.set("is_open", False)
        self.context.set("trade", self.trade)

    def apply_action(self, action):
        """The apply_action method applied action, calculating reward and open/close trade"""
        self.context.set("action", action)
        reward, action_result = getattr(self, self.router[action])()
        self.context.set("reward", reward)
        return reward, action_result

    def apply_action_wait(self):
        raise NotImplementedError

    def apply_action_open(self):
        raise NotImplementedError

    def apply_action_hold(self):
        raise NotImplementedError

    def apply_action_close(self):
        raise NotImplementedError

    def _get_penalty(self, val=None):
        """Расчет штрафа. Если штраф не задан явно, то берем из базового значения"""
        value = self.penalty if val is None else val
        return value
