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
        0: "_action_wait",
        1: "_action_open",
        2: "_action_hold",
        3: "_action_close"
    }

    def __init__(self, context=None, penalty=-2, reward=0):
        self.context = context
        self.penalty = penalty
        self.reward = reward

        self.trade = None
        self.reset()
        print("BaseActionRouter")

    def reset(self):
        """Reset current action controller state"""
        self.trade = None
        self.context.set("is_open", False)
        self.context.set("trade", self.trade)

    def apply_action(self, action):
        """The apply_action method applied action, calculating reward and open/close trade"""
        is_open = self.context.get("is_open")
        ts = self.context.get("ts")
        handler = getattr(self, self.router[action])
        reward, action_result = handler(ts, is_open)
        return reward, action_result

    def _get_penalty(self, val=None):
        """Расчет штрафа. Если штрафне задан явно, то берем из базового значения"""
        value = self.penalty if val is None else val
        return value

    def _action_wait(self, ts, is_open):
        raise NotImplementedError

    def _action_open(self, ts, is_open):
        raise NotImplementedError

    def _action_hold(self, ts, is_open):
        raise NotImplementedError

    def _action_close(self, ts, is_open):
        raise NotImplementedError
