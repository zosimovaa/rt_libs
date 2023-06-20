"""
Gym trade stock environment
"""

import logging
import numpy as np
import gym

from .log_setup import logger_setup


class TradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, core, dp_factory, alias="test run", log=True, log_obs=False):
        super().__init__()
        self.alias = alias
        self.core = core
        self.dp_factory = dp_factory
        self.log = log
        self.log_obs = log_obs

        self.logger = logging.getLogger(__name__)
        self.logger = logger_setup(self.logger, self.alias)

        self.episode = -1
        self.step_num = 0
        self.step_info = {}

        data_point = self.dp_factory.reset()
        self.core.reset(data_point=data_point)

        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        self.logger.info("Observation space {}".format(self.observation_space))

    def get_action_space(self):
        return self.core.get_action_space()

    def get_observation_space(self):
        data_point = self.dp_factory.get_current_step()
        observation = self.core.get_observation(data_point)

        if isinstance(observation, list):
            observation_space = [inp.shape for inp in observation]
        else:
            observation_space = observation.shape
        return observation_space

    def reset(self):
        self.step_num = 0
        self.episode += 1

        data_point = self.dp_factory.reset()
        self.core.reset(data_point=data_point)
        observation = self.core.get_observation(data_point)
        return observation

    def step(self, action):
        reward, action_result = self.core.apply_action(action)

        self.render()

        data_point, done = self.dp_factory.get_next_step()
        observation = self.core.get_observation(data_point)

        self.step_num = self.step_num + 1

        return observation, reward, done, self.step_info

    def render(self, mode='ansi'):
        if self.log:
            self.step_info = self.get_step_info()
            message = "Cursor: {cursor:<5} | Action: {action:<3} -> | State: {state:<2} -> | " \
                      "Reward: {reward:>8.3f} | Profit: {profit:>8.3f} | TotReward: {total_reward:>9.3f} | " \
                      "Balance: {balance:>8.3f} | lowest_ask: {lowest_ask:>9.3f} | highest_bid: {highest_bid:>9.3f} |" \
                      " |---| {observation}"
            message = message.format(**self.step_info)
            self.logger.warning(message)

    def log_episode_result(self, metrics):
        """Метод записывает данные в лог для оффлайн лог ридера"""
        message = ";".join(["{" + val + "}" for val in metrics.keys()])
        message = message.format(**metrics)
        self.logger.warning(message)

    def get_step_info(self):
        """Метод записывает данные в лог для детального разбора того, что происходит"""
        if self.log_obs:
            obs = obs_to_string(self.core.context.get("observation"))
        else:
            obs = None

        step_info = {
            "cursor": self.core.context.get("ts"),
            "state": self.core.context.get("is_open"),
            "observation": obs,
            "action": self.core.context.get("action"),
            "reward": self.core.context.get("reward"),
            "total_reward": self.core.metric_collector.get_metric("TotalReward"),
            "balance": self.core.metric_collector.get_metric("Balance"),
            "profit": self.core.context.get("profit"),
            "lowest_ask": self.core.context.get("lowest_ask"),
            "highest_bid": self.core.context.get("highest_bid")
        }
        return step_info

def obs_to_string(observation):
    if isinstance(observation, list):
        obs_formatted = []
        for obs in observation:
            obs_formatted.append(np.array2string(obs, max_line_width=500, precision=8, separator=',',
                                                 suppress_small=True).replace("\n", " | "))
        obs_formatted = "; ".join(map(str, obs_formatted))
    else:
        obs_formatted = np.array2string(observation, max_line_width=500, precision=8, separator=',',
                                        suppress_small=True).replace("\n", " | ")
    return obs_formatted
