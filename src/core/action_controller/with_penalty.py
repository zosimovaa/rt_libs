from .basic import ActionControllerBasic, ActionControllerBasicOpposite

class ActionControllerFixedPenalty(ActionControllerBasic):
    def _action_hold(self, ts, is_open):
        if is_open:
            reward = self.penalty * self.scale_hold
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result


class ActionControllerIncreasedPenalty(ActionControllerBasic):
    def _action_hold(self, ts, is_open):
        raise NotImplementedError
