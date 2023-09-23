"""
Тест базового тикера
Награда только при закрытии
"""

import unittest
import numpy as np

from tests.test_dataset import TestDatasetGenerator

from src.data_point import DataPointFactory
from src.core_v1.context import Context
from src.core_v1.actions import TradeAction, BadAction
from src.core_v1.action_controller import ActionControllerDiffReward


class TickerBasicTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_observation_points = 5
        self.market_fee = 0.005
        self.dataset = TestDatasetGenerator().make(
            num=100,
            lowest_ask=20000,
            highest_bid=19900,
            ts_start=1000,
            price_step=100,
            period=60
        )

        self.dpf = DataPointFactory(
            self.dataset,
            period=60,
            n_observation_points=self.n_observation_points,
            n_history_points=0,
            n_future_points=0,
            step_size=None
        )
        self.context = Context(market_fee=self.market_fee)

        self.reward = 0
        self.penalty = -2

    def test_action_wait(self):
        ticker = ActionControllerDiffReward(self.context, penalty=self.penalty, reward=self.reward)
        ticker.scale_wait = 1
        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        ticker.reset()

        # Case 1 - Позиция закрыта
        reward, action_result = ticker.apply_action(0)
        reward_exp = -ticker._get_diff_reward()
        self.assertEqual(action_result, None)
        self.assertEqual(reward, reward_exp)

        # Case 2 - Следующий шаг
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        reward, action_result = ticker.apply_action(0)
        reward_exp = -ticker._get_diff_reward()
        self.assertEqual(action_result, None)
        self.assertEqual(reward, reward_exp)

        # Case 3 - Влияние scale
        ticker.scale_wait = 0
        reward, action_result = ticker.apply_action(0)
        reward_exp = 0
        self.assertEqual(action_result, None)
        self.assertEqual(reward, reward_exp)

        # Case 4 - Позиция открыта
        ticker.scale_wait = 1
        reward, action_result = ticker.apply_action(1)
        reward, action_result = ticker.apply_action(0)
        reward_exp = ticker.penalty
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(reward, reward_exp)

        # Case 5 - Следующий шаг
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)
        reward, action_result = ticker.apply_action(0)
        reward_exp = ticker.penalty
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(reward, reward_exp)

        # Case 6 - Влияние scale
        ticker.scale_wait = 0
        reward, action_result = ticker.apply_action(0)
        reward_exp = ticker.penalty
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(reward, reward_exp)


    def test_action_open(self):
        # Позиция закрыта
        # Следующий шаг
        # Влияние scale
        # Позиция открыта
        # Следующий шаг
        # Влияние scale
        pass

    def test_action_hold(self):
        # Позиция закрыта
        # Следующий шаг
        # Влияние scale
        # Позиция открыта
        # Следующий шаг
        # Влияние scale
        pass

    def test_action_close(self):
        # Позиция закрыта
        # Следующий шаг
        # Влияние scale
        # Позиция открыта
        # Следующий шаг
        # Влияние scale
        pass

    def test_get_diff_reward(self):
        pass



    def closed(self):

        # ==== Step 1 - подготовка тикера ====
        ticker = ActionControllerDiffReward(self.context, penalty=self.penalty, reward=self.reward)

        ticker.scale_open = 1
        ticker.scale_close = 1

        ticker.reset()

        # ==== Step 2 - тест без торговой операции ====
        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)

        # check trade status before start
        is_open = self.context.get("is_open", domain="Trade")
        self.assertEqual(is_open, False)

        # action = 0
        reward, action_result = ticker.apply_action(0)
        self.assertEqual(action_result, None)
        self.assertEqual(reward, 0)

        # action = 2
        reward, action_result = ticker.apply_action(2)
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(np.round(reward, 3), np.round(self.penalty, 3))

        # action = 3
        reward, action_result = ticker.apply_action(3)
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(np.round(reward, 3), np.round(self.penalty, 3))

        # action = 1
        reward, action_result = ticker.apply_action(1)
        reward_exp = 0 #self.context.get("highest_bid") / self.context.get("lowest_ask") - 1 - self.context.market_fee
        self.assertEqual(isinstance(action_result, TradeAction), True)
        self.assertEqual(np.round(reward, 3), np.round(reward_exp, 3))

        # Trade opened, next datapoint
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        # action = 0
        reward, action_result = ticker.apply_action(0)
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(np.round(reward, 3), np.round(self.penalty, 3))

        # action = 1
        reward, action_result = ticker.apply_action(1)
        reward_exp = self.context.get("highest_bid") / self.context.get("lowest_ask") - 1 - self.context.market_fee
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(np.round(reward, 3), np.round(self.penalty, 3))

        # action = 2
        reward, action_result = ticker.apply_action(2)
        self.assertEqual(action_result, None)
        self.assertEqual(reward, 0)


    def opened(self):

        # ==== Step 1 - подготовка тикера ====
        ticker = ActionControllerDiffReward(self.context, penalty=self.penalty, reward=self.reward)

        ticker.scale_open = 1
        ticker.scale_close = 1

        ticker.reset()

        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        reward, action_result = ticker.apply_action(1)

        # check trade status before start
        is_open = self.context.get("is_open", domain="Trade")
        self.assertEqual(is_open, True)



        # action = 3
        reward, action_result = ticker.apply_action(3)
        reward_exp = self.context.get("highest_bid") / self.context.trade.open_price - 1 - self.context.market_fee

        self.assertEqual(isinstance(action_result, TradeAction), True)
        self.assertEqual(np.round(reward, 3), np.round(reward_exp, 3))


if __name__ == '__main__':
    unittest.main()

