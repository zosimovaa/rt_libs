"""
Тест тикера с расширенной наградой
Пререквизиты - контекст, торговая операция
"""

import unittest
import numpy as np

from tests.test_dataset import TestDatasetGenerator

from src.data_point import DataPointFactory
from src.core.context import BasicContext
from src.core.actions import TradeAction, BadAction
from src.core.tickers import TickerOppositeTradesReward
from src.core.observation_builder.features import TradeStateFeature, ProfitFeature, OppositeProfitFeature
from src.core.observation_builder import ObservationBuilder2Dim

from src.core.facade import RTCore


class TickerBasicTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_observation_points = 5

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

        self.market_fee = 0.005
        self.reward = 0
        self.penalty = -2

        self.context = BasicContext(market_fee=self.market_fee)
        self.ticker = TickerOppositeTradesReward(self.context, penalty=self.penalty, reward=self.reward)
        self.ticker.REWARD_OPEN = 1
        self.ticker.REWARD_CLOSE = 1
        self.ticker.REWARD_WAIT = 1
        self.ticker.REWARD_HOLD = 1
        self.ticker.NUM_MEAN_OBS = 2

        trade_state_feat = TradeStateFeature(self.context)
        profit_feat = ProfitFeature(self.context, scale_factor=1)
        opposite_profit_feat = OppositeProfitFeature(self.context, scale_factor=1)
        self.observation_builder = ObservationBuilder2Dim(self.context, [trade_state_feat], [profit_feat, opposite_profit_feat ])

        self.core = RTCore(self.context, self.ticker, self.observation_builder)


    def test_reward(self):
        # ==== Инициализация ====
        data_point = self.dpf.get_current_step()
        self.core.reset(data_point=data_point)

        # 1. Action=0.
        observation = self.core.get_observation(data_point=data_point)
        # print(observation)

        reward, action_result = self.core.apply_action(0)
        rates_diff_mean = np.mean(self.core.action_controller.get_last_diffs())
        reward_exp = -rates_diff_mean / self.context.get("highest_bid") * self.ticker.REWARD_WAIT

        self.assertEqual(np.round(reward, 5), np.round(reward_exp, 5))


        # 2. Action=1. Открываем операцию.
        data_point, done = self.dpf.get_next_step()
        observation = self.core.get_observation(data_point=data_point)

        reward, action_result = self.core.apply_action(1)
        ot = self.core.context.get("trade", domain="OppositeTrade")
        reward_exp = -ot.get_profit()

        self.assertEqual(np.round(reward, 5), np.round(reward_exp, 5))

        Доделать








if __name__ == '__main__':
    unittest.main()

