"""
Модуль содержит реализации ReplayBuffer для применения в алгоритме DQN
Версия ReplayBufferDumb работает быстрее на небольших объемах.
С ростом количества фреймов больше 200к ситуация меняется - ReplayBuffer начинает работать быстрее.
"""
import random
import numpy as np
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *params):
        self.buffer.append(params)

    def sample(self, batch_size):
        return map(np.array, zip(*random.sample(self.buffer, batch_size)))

    def __len__(self):
        return len(self.buffer)


class ReplayBufferDumb(object):
    def __init__(self, capacity):
        self.max_memory_length = capacity
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []

    def push(self, state, action, reward, next_state, done):
        self.state_history.append(state)
        self.action_history.append(action)
        self.rewards_history.append(reward)
        self.state_next_history.append(next_state)
        self.done_history.append(done)

        if len(self.done_history) > self.max_memory_length:
            del self.rewards_history[0]
            del self.state_history[0]
            del self.state_next_history[0]
            del self.action_history[0]
            del self.done_history[0]

    def sample(self, batch_size):
        indices = random.sample(range(len(self.done_history)), batch_size)
        state_sample = [self.state_history[i] for i in indices]
        next_state_sample = [self.state_next_history[i] for i in indices]
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = 1 * np.array([self.done_history[i] for i in indices])

        return state_sample, action_sample, rewards_sample, next_state_sample, done_sample

    def __len__(self):
        return len(self.action_history)


