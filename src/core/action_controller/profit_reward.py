from .basic import ActionControllerBasic, ActionControllerBasicOpposite

class ActionControllerProfitReward(ActionControllerBasicOpposite):
    """Класс реализует логику расчета награды/штрафа за действия и профита за торговые операции"
    Помимо награзы в виде профита за открытие/закрытие добавляется награда в ожидании в виде изменения курса.
    """
    def _action_wait(self, ts, is_open):
        if is_open:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        else:
            opposite_trade = self.context.get("trade", domain="OppositeTrade")
            reward = -opposite_trade.get_profit() * self.scale_wait
            action_result = None
        return reward, action_result

    def _action_hold(self, ts, is_open):
        if is_open:
            profit = self.context.get("profit", domain="Trade")
            reward = profit * self.scale_hold
            action_result = None
        else:
            reward = self._get_penalty()
            action_result = BadAction(self.context)
        return reward, action_result

    def _get_last_diffs(self, column='lowest_ask'):
        data_point = self.context.data_point
        num = self.num_mean_obs + 1
        feature_values = data_point.get_values(column, num=num)
        result = np.diff(feature_values)
        return result