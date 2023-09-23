"""
Тест тикера с расширенной наградой
Пререквизиты - контекст, торговая операция
"""

import unittest
import numpy as np

from tests.test_dataset import TestDatasetGenerator

from src.data_point import DataPointFactory
from src.core_v1.context import BasicContext
from src.core_v1.actions import TradeAction, BadAction
from src.core_v1.tickers import TickerWaitHoldDiff


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
        ticker = TickerWaitHoldDiff(self.context, penalty=self.penalty, reward=self.reward)

        ticker.REWARD_OPEN = 1
        ticker.REWARD_CLOSE = 1
        ticker.reward_wait = 1
        ticker.reward_hold = 1
        ticker.last_points_mean = 2

        ticker.reset()

        # ==== Step 2 - тест без торговой операции ====
        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)

        is_open = self.context.get("is_open", domain="Trade")
        self.assertEqual(is_open, False)

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
        reward, action_result = ticker.apply_action(1)
        reward_exp = self.context.get("highest_bid") / self.context.get("lowest_ask") - 1 - self.context.market_fee
        self.assertEqual(isinstance(action_result, TradeAction), True)
        self.assertEqual(np.round(reward, 3), np.round(reward_exp, 3))


    def test_opened(self):

        # ==== Step 1 - подготовка тикера ====
        ticker = TickerWaitHoldDiff(self.context, penalty=self.penalty, reward=self.reward)

        ticker.REWARD_OPEN = 1
        ticker.REWARD_CLOSE = 1
        ticker.last_points_mean = 2

        ticker.reset()

        data_point = self.dpf.get_current_step()
        self.context.update_datapoint(data_point)
        reward, action_result = ticker.apply_action(1)


        is_open = self.context.get("is_open", domain="Trade")
        self.assertEqual(is_open, True)

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

