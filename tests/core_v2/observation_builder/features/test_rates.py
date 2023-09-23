import unittest
import numpy as np

from tests.test_dataset import TestDatasetGenerator_v2
from src.core_v2.data_point import DataPointFactory, DataPoint
from src.core_v2.context import Context
from src.core_v2.observation_builder.features import RatesFeature


class RatesFeatureTestCase(unittest.TestCase):
    ALIAS = "unittests"
    OBSERVATION_LEN = 5
    STEP_FACTORS = (1, 2, 5)

    DF_NUM = 50
    DF_FEATURES = {
        "lowest_ask": {"start": 101, "step": 1.01},
        "highest_bid": {"start": 100, "step": 1}
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"= = = = = = Start {self.__class__.__name__} = = = = = =")

        # test params
        self.context = Context(alias=self.ALIAS)
        self.dpf = DataPointFactory(
            TestDatasetGenerator_v2.make(self.DF_FEATURES, num=self.DF_NUM),
            step_size=1,
            offset=max(self.STEP_FACTORS) * self.OBSERVATION_LEN,
            observation_len=self.OBSERVATION_LEN,
            future_points=0
        )


    def test_feature(self):
        self.dpf.reset()
        dp = self.dpf.get_current_step()
        self.context.set_dp(dp)

        for sf in self.STEP_FACTORS:
            feature = RatesFeature(alias=self.ALIAS, step_factor=sf, scale_output=1)
            observed = feature.get()

            current_value = dp.data["highest_bid"].values[-1]
            values = dp.get_values("highest_bid", step_factor=sf)
            expected = values / current_value - 1
            expected = np.average(expected.reshape(-1, sf), axis=1)

            for i in range(len(expected)):
                self.assertEqual(np.round(observed[i], 5), np.round(expected[i], 5))

if __name__ == '__main__':
    unittest.main()
