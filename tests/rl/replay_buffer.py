import unittest

import time
import numpy as np

from rl.components import ReplayBufferDumb, ReplayBuffer


class PerfomanceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = 32
        self.buf_size = 100000
        self.frames = 400000

    def test_common(self):
        rb = ReplayBuffer(self.buf_size)

        state = np.array(range(100)).reshape(10, 10)
        action = 0
        reward = 1
        done = False

        start = time.time()
        for i in range(self.frames):
            rb.push(state, action, reward, state, done)

            if len(rb) >= self.batch_size:
                states, actions, rewards, next_states, dones = rb.sample(self.batch_size)
        end = time.time()
        duration = end - start
        print(f"ReplayBuffer duration: {duration:<4.2f}")

        self.assertEqual(True, True)

    def test_fast(self):
        rb = ReplayBufferDumb(self.buf_size)

        state = np.array(range(100)).reshape(10, 10)
        action = 0
        reward = 1
        done = False

        start = time.time()
        for i in range(self.frames):
            rb.push(state, action, reward, state, done)

            if len(rb) >= self.batch_size:
                states, actions, rewards, next_states, dones = rb.sample(self.batch_size)
        end = time.time()
        duration = end - start
        print(f"ReplayBufferDumb duration: {duration:<4.2f}")

        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
