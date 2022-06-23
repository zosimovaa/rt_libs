import unittest
from src.core.actions import TradeAction

INIT_TEST_DATA = {
    "ts": 600,
    "lowest_ask": 100,
    "highest_bid": 101,
    "market_fee": 0.005
}


class TradeActionTestCase(unittest.TestCase):
    def dtest_init(self):
        trade_action = TradeAction(**INIT_TEST_DATA)
        self.assertEqual(trade_action.profit, 0.005)  # add assertion here
        self.assertEqual(trade_action.is_open, True)

    def dtest_close(self):
        trade_action = TradeAction(**INIT_TEST_DATA)
        ts = 660
        highest_bid = 102
        self.assertEqual(trade_action.is_open, True)
        trade_action.close(ts, highest_bid)
        self.assertEqual(trade_action.profit, 0.015)  # add assertion here
        self.assertEqual(trade_action.is_open, False)

        highest_bid = 103
        trade_action.update(highest_bid)

        self.assertEqual(trade_action.profit, 0.015)  # add assertion here
        self.assertEqual(trade_action.get_profit(), 0.)  # add assertion here


if __name__ == '__main__':
    unittest.main()
