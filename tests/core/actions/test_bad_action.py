import unittest
from src.core.actions import BadAction


class BadActionTestCase(unittest.TestCase):
    def test_init(self):
        ts = 100000
        action = "open_trade"

        bad_action = BadAction(ts, action)

        self.assertEqual(bad_action.ts, ts)
        self.assertEqual(bad_action.action, action)

if __name__ == '__main__':
    unittest.main()
