"""
Сложный контроллер со следующей логики формирования награды
 - wait: сложная функция
    (нужно иметь ввиду что положительный профит от TradeOpposite - это плохо!)
    - при нулевом и положительном профите от trade_opposite - есть 2 варианта: не возвращать ничего или возвращать только инвертированные положительные
      (т.е. со знаком минус) награды между точками
    - при отрицательном профите от trade_opposite - возвращаем любую награду от разницы между точками
 - open: профит от trade_opposite
 - hold: сложная функция
    - при нулевом и положительном профите: есть 2 варианта - не возвращать иничего или возвращть любую награду от разницы между точками
    - при отрицательном профите только негативные изменения между точками.
 - close: профит от trade

Уровень награды можно регулировать через коэффициенты xxxx_scale (в том числе и отключать)

Суть такого контроллера со сложной наградой - заставить агента опасаться просадок, когда профит уже в минусе.
С diff_reward подходом была проблема, что агент держит открытой операцию, когда курс сильно просел, т.к. в моменте нет
сильного штрафа и если есть выбор между "не получать штраф вообще (не закрывать позицию)" и закрыть с минусом -
агент выберет первый вариант. На синтетических данных этот эффектом получилось устранить повышением награды за
wait и hold, но на реальных данных много шума и такая тактика не вообще работает.

"""

import numpy as np
from .basic import BasicTrainController
from core_v2.actions import BadAction, VoidAction, TradeAction


class TrainControllerProfitStateBalanced(BasicTrainController):
    """Для WAIT и HOLD при отрицательном профите будем давать только отрицательную награду, пири положительно - только положительную"""

    def __init__(self, *args, **kwargs):
        self.edge = kwargs.pop("edge", 0)
        super().__init__(*args, **kwargs)

    def apply_action_wait(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        else:
            result_action = VoidAction(ts, is_open)
            if self.wait_scale:
                reward = -1 * self._get_wh_reward() * self.wait_scale

                price = self.context.get("highest_bid")
                profit_opposite = self.trade_opposite.get_profit(price)

                # Если профит во время ожидания положительный - это плохо - теряем возможности.
                # Тогда надо штрафовать - даем только отрицательную награду.
                if profit_opposite > self.edge:
                    reward = min(0, reward)
                # Если профит во время ожидания отрицаельный - это хорошо - выжидаем наблагоприятный период.
                # Тогда надо давать положительную награду.
                elif profit_opposite < -self.edge:
                    reward = max(0, reward)
                else:
                    pass
            else:
                reward = 0

        return reward, result_action

    def apply_action_hold(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")

        if is_open:
            result_action = VoidAction(ts, is_open)
            if self.hold_scale:

                reward = self._get_wh_reward() * self.hold_scale
                price = self.context.get("highest_bid")
                profit = self.trade.get_profit(price)

                if profit > self.edge:
                    reward = max(0, reward)
                elif profit < -self.edge:
                    reward = min(0, reward)
                else:
                    pass

            else:
                reward = 0
        else:
            reward = self.penalty
            result_action = BadAction(ts, is_open)

        return reward, result_action


class TrainControllerProfitStateNPNR(BasicTrainController):
    """Для WAIT и HOLD при отрицательном профите будем давать только отрицательную награду"""

    def __init__(self, *args, **kwargs):
        self.edge = kwargs.pop("edge", 0)
        super().__init__(*args, **kwargs)


    def apply_action_wait(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")
        if is_open:
            reward = self.penalty
            result_action = BadAction(ts, is_open)
        else:
            result_action = VoidAction(ts, is_open)
            if self.wait_scale:
                reward = -1 * self._get_wh_reward() * self.wait_scale

                price = self.context.get("highest_bid")
                profit_opposite = self.trade_opposite.get_profit(price)

                if profit_opposite > self.edge:
                    reward = min(0, reward)
            else:
                reward = 0

        return reward, result_action

    def apply_action_hold(self):
        ts = self.context.get("ts")
        is_open = self.context.get("is_open")

        if is_open:
            result_action = VoidAction(ts, is_open)
            if self.hold_scale:

                reward = self._get_wh_reward() * self.hold_scale
                price = self.context.get("highest_bid")
                profit = self.trade.get_profit(price)

                if profit < -self.edge:
                    reward = min(0, reward)

            else:
                reward = 0
        else:
            reward = self.penalty
            result_action = BadAction(ts, is_open)

        return reward, result_action
