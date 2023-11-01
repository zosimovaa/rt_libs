"""
Тест тикера с оппозитной торговой операцией
Пререквизиты - контекст, торговая операция
"""

import unittest
import numpy as np

from tests.test_dataset import TestDatasetGenerator

from core_v1.data_point import DataPointFactory
from src.core_v1.context import BasicContext
from src.core_v1.actions import TradeAction, BadAction


class TickerBasicTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_observation_points = 5
        self.market_fee = 0.005
        self.dataset = TestDatasetGenerator().make(
            num=100,
            lowest_ask=100,
            highest_bid=99,
            ts_start=1000,
            price_step=1,
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
        self.context = BasicContext(market_fee=self.market_fee)

        self.reward = 0
        self.penalty = -2

    def test_closed(self):
        # ==== Step 1 - подготовка тикера ====
        ticker = TickerOppositeTradesReward(self.context, penalty=self.penalty, reward=self.reward)
        ticker.reward_open = 1
        ticker.reward_close = 1
        ticker.reward_wait = 1
        ticker.reward_hold = 1
        ticker.last_points_mean = 2

        # ==== Step 2 - тест без торговой операции ====
        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        ticker.reset()

        open_price = self.context.get("highest_bid")

        is_open = self.context.get("is_open", domain="Trade")
        self.assertEqual(is_open, False)

        # проверки состояния OppositeTrade
        ot = self.context.get("trade", domain="OppositeTrade")
        self.assertEqual(ot.is_open, True)
        profit = ot.get_profit()
        self.assertEqual(profit, 0)


        # action = 0
        reward, action_result = ticker.apply_action(0)
        reward_exp = - np.mean(ticker.get_last_diffs()) / self.context.get("highest_bid") * ticker.reward_wait
        self.assertEqual(action_result, None)
        self.assertEqual(np.round(reward, 4), np.round(reward_exp, 4))

        # action = 2
        reward, action_result = ticker.apply_action(2)
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(np.round(reward, 3), np.round(self.penalty, 3))

        # action = 3
        reward, action_result = ticker.apply_action(3)
        self.assertEqual(isinstance(action_result, BadAction), True)
        self.assertEqual(np.round(reward, 3), np.round(self.penalty, 3))



        # action = 1
        data_point, done = self.dpf.get_next_step()
        data_point, done = self.dpf.get_next_step()
        data_point, done = self.dpf.get_next_step()
        data_point, done = self.dpf.get_next_step()
        self.context.update_datapoint(data_point)

        reward, action_result = ticker.apply_action(1)
        close_price = self.context.get("highest_bid")

        reward_exp = -(close_price / open_price - 1)
        self.assertEqual(isinstance(action_result, TradeAction), True)
        self.assertEqual(np.round(reward, 5), np.round(reward_exp, 5))





    def test_opened(self):

        # ==== Step 1 - подготовка тикера ====
        ticker = TickerOppositeTradesReward(self.context, penalty=self.penalty, reward=self.reward)

        ticker.reward_open = 1
        ticker.reward_close = 1
        ticker.last_points_mean = 2

        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        ticker.reset()

        reward, action_result = ticker.apply_action(1)

        is_open = self.context.get("is_open", domain="Trade")
        self.assertEqual(is_open, True)
        ot = self.context.get("trade", domain="OppositeTrade")
        self.assertEqual(ot.is_open, False)

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
        reward_exp = np.mean(ticker.get_last_diffs()) / self.context.get("highest_bid") * ticker.reward_wait
        self.assertEqual(action_result, None)
        self.assertEqual(np.round(reward, 4), np.round(reward_exp, 4))

        # action = 3
        reward, action_result = ticker.apply_action(3)
        reward_exp = self.context.get("highest_bid") / self.context.trade.open_price - 1 - self.context.market_fee

        self.assertEqual(isinstance(action_result, TradeAction), True)
        self.assertEqual(np.round(reward, 3), np.round(reward_exp, 3))

if __name__ == '__main__':
    unittest.main()

