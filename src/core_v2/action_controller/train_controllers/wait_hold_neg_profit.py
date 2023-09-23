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


class TrainControllerWaitHoldNegProfit(BasicTrainController):
    """Для WAIT и HOLD при отрицательном профите будем давать только отрицательную награду"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_action_wait(self):
        is_open = self.context.get("is_open")
        if is_open:
            return self.penalty
        else:
            if self.wait_scale:
                # минуc добавляется т.к. при росте курса в отсутствии открытой операции нужно дать штраф.
                reward = -1 * self._get_wh_reward() * self.wait_scale

                trade_opposite = self.context.get("trade_opposite")
                price = self.context.get("highest_bid")
                profit_opposite = trade_opposite.get_profit(price)

                if profit_opposite > 0:
                    reward = min(0, reward)
            else:
                reward = 0
            return reward

    def apply_action_hold(self):
        is_open = self.context.get("is_open")
        if not is_open:
            return self.penalty
        else:
            if self.wait_scale:
                reward = 0
            else:
                reward = self._get_wh_reward() * self.wait_scale

                trade = self.context.get("trade")
                price = self.context.get("highest_bid")
                profit = trade.get_profit(price)

                if profit < 0:
                    reward = min(0, reward)

            return reward

