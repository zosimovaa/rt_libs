import unittest
import numpy as np
from collections import deque

from tests.test_dataset import TestDatasetGenerator_v2
from src.data_point import DataPointFactory, DataPoint
from src.core_v2.context import Context
from src.core_v2.observation_builder.features import ProfitFeature
from src.core_v2.action_controller.trade_controllers import TrainController4Actions


class ProfitFeatureTestCase(unittest.TestCase):
    ALIAS = "unittests"
    OBSERVATION_LEN = 5
    STEP_FACTORS = (1,)

    DF_NUM = 50
    DF_FEATURES = {
        "lowest_ask": {"start": 101, "step": 1.01},
        "highest_bid": {"start": 100, "step": 1}
    }

    ASSERT_PRESISION = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"= = = = = = Start {self.__class__.__name__} = = = = = =")

        # test params
        self.context = Context(alias=self.ALIAS)
        self.trade_ctrl = TrainController4Actions(alias=self.ALIAS, market_fee=0)
        self.dpf = DataPointFactory(
            TestDatasetGenerator_v2.make(self.DF_FEATURES, num=self.DF_NUM),
            step_size=1,
            offset=max(self.STEP_FACTORS) * self.OBSERVATION_LEN,
            observation_len=self.OBSERVATION_LEN,
            future_points=0
        )

    def check_results(self, observed, expected):
        self.assertEqual(len(observed), len(expected))
        for i in range(len(expected)):
            self.assertEqual(np.round(observed[i], self.ASSERT_PRESISION), np.round(expected[i], self.ASSERT_PRESISION))


    def test_calculation(self):
        self.dpf.reset()

        for sf in self.STEP_FACTORS:

            # Step 1.1 - trade closed
            dp = self.dpf.get_current_step()
            self.context.set_dp(dp)

            self.trade_ctrl.apply_action_close()
            feature = ProfitFeature(alias=self.ALIAS, step_factor=sf, scale_output=1)
            observed = feature.get()

            expected = np.zeros(dp.observation_len)

            self.check_results(observed, expected)

            # Step 1.2 - trade closed, text dp
            dp, done = self.dpf.get_next_step()
            self.context.set_dp(dp)

            self.trade_ctrl.apply_action_close()
            observed = feature.get()
            expected = deque([0]*self.OBSERVATION_LEN, maxlen=self.OBSERVATION_LEN)
            self.check_results(observed, expected)

            # Step 2.1 - trade opened

            self.trade_ctrl.apply_action_open()
            open_price = self.context.get("lowest_ask")

            for i in range(self.OBSERVATION_LEN):
                observed = feature.get()
                h_bid = self.context.get("highest_bid")
                expected.append(h_bid/open_price-1)
                self.check_results(observed, expected)

                dp, done = self.dpf.get_next_step()
                self.context.set_dp(dp)

            # Step 2.2 - trade close
            self.trade_ctrl.apply_action_close()
            observed = feature.get()
            expected = np.zeros(dp.observation_len)
            self.check_results(observed, expected)



if __name__ == '__main__':
    unittest.main()
